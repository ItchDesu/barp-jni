use std::net::{Ipv4Addr, Ipv6Addr};
use std::os::fd::{AsRawFd, FromRawFd, RawFd};
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicI64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{anyhow, Result};
use jni::objects::{JClass, JString};
use jni::sys::{jboolean, jint, jobject};
use jni::JNIEnv;
use log::info;
use mtproto_vpn::crypto_dh;
use mtproto_vpn::crypto_ige;
use mtproto_vpn::rpc;
use mtproto_vpn::transport::TransportReader;
use num_bigint::BigUint;
use rand::Rng;
use sha1::{Digest, Sha1};
use sha2::Sha256 as Sha256Hash;
use tokio::io::unix::AsyncFd;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::net::TcpStream;
use mtproto_vpn::faketls::{self, FakeTlsReader, FakeTlsWriter, ProxySecret};
use tokio::runtime::Runtime;
use tokio::sync::mpsc;

use grammers_tl_types::deserialize::{Cursor, Deserializable};
use grammers_tl_types::{functions, types, enums, Serializable, Identifiable};
use rsa::pkcs1::DecodeRsaPublicKey;
use rsa::traits::PublicKeyParts;

// Server configuration is embedded in the native library, just like Telegram
// hard-codes datacenter addresses and public keys in tgnet.
const SERVER_HOST: &str = "5.188.183.132";
const SERVER_PORT: u16 = 20443;

// Optional MTProto proxy (anti-censorship), configured at runtime from the app
// (like Telegram's proxy settings) via `setProxyNative`. When set, the tunnel is
// routed through a proxy that forwards to SERVER_HOST. The secret accepts the
// same encodings Telegram uses: `dd<hex>` for padded obfuscation, or
// `ee<hex><domain-hex>` (or `ee<hex>:example.com`) for FakeTLS, which disguises
// the connection as HTTPS to the given fronting domain. When empty the client
// connects to SERVER_HOST directly with plain obfuscation.
static PROXY_ADDR: Mutex<Option<String>> = Mutex::new(None);
static PROXY_SECRET: Mutex<Option<String>> = Mutex::new(None);

/// Set (or clear) the proxy configuration. Empty strings clear it.
fn set_proxy(addr: Option<String>, secret: Option<String>) {
    *PROXY_ADDR.lock().unwrap() = addr.filter(|s| !s.trim().is_empty());
    *PROXY_SECRET.lock().unwrap() = secret.filter(|s| !s.trim().is_empty());
}

/// The parsed proxy secret, if a proxy is configured.
fn proxy_secret() -> Result<Option<ProxySecret>> {
    let addr = PROXY_ADDR.lock().unwrap().clone();
    let secret = PROXY_SECRET.lock().unwrap().clone();
    match (addr, secret) {
        (Some(_), Some(s)) => Ok(Some(
            ProxySecret::parse(&s).map_err(|e| anyhow!("invalid proxy secret: {}", e))?,
        )),
        (None, None) => Ok(None),
        _ => Err(anyhow!("proxy address and secret must both be set to use a proxy")),
    }
}

/// The TCP endpoint the tunnel connects to: the proxy when configured, else the
/// VPN server. Returns an IPv4 address and port (host must be a literal IPv4).
fn tunnel_target() -> Result<(Ipv4Addr, u16)> {
    let addr = PROXY_ADDR.lock().unwrap().clone();
    let has_secret = PROXY_SECRET.lock().unwrap().is_some();
    match addr {
        Some(addr) if has_secret => {
            let (host, port) = addr
                .rsplit_once(':')
                .ok_or_else(|| anyhow!("proxy address must be host:port"))?;
            Ok((host.parse()?, port.parse()?))
        }
        _ => Ok((SERVER_HOST.parse()?, SERVER_PORT)),
    }
}

static RUNTIME: OnceLock<Runtime> = OnceLock::new();

struct GlobalState {
    session: Option<Arc<VpnSession>>,
    assigned_ipv4: Option<Ipv4Addr>,
    assigned_ipv6: Option<Ipv6Addr>,
    write_tx: Option<mpsc::Sender<Vec<u8>>>,
    // Read side of the connection, handed to the bridge so it can receive
    // server->client traffic. Taken out (Option::take) when the bridge starts.
    transport_reader: Option<TransportReader>,
    tcp_reader: Option<BoxRead>,
}

struct VpnSession {
    auth_key: [u8; 256],
    auth_key_id: i64,
    salt: i64,
    session_id: [u8; 8],
    server_seq_no: Mutex<i32>,
}

static STATE: Mutex<GlobalState> = Mutex::new(GlobalState {
    session: None,
    assigned_ipv4: None,
    assigned_ipv6: None,
    write_tx: None,
    transport_reader: None,
    tcp_reader: None,
});
static CONNECTED: AtomicBool = AtomicBool::new(false);

/// TCP socket created by `prepareSocketNative` and handed to Java so
/// `VpnService.protect()` can exempt it from the VPN routes before it is
/// connected. Without this, once the VPN comes up with a default route the
/// tunnel's own TCP packets are routed back into the TUN device and the
/// connection feeds on itself (VPN "connected" but zero traffic flows).
static PREPARED_SOCKET: AtomicI32 = AtomicI32::new(-1);

/// Handle of the running bridge task, so disconnect can abort it and release
/// the tun fd and the TCP read half instead of leaking them.
static BRIDGE_TASK: Mutex<Option<tokio::task::JoinHandle<()>>> = Mutex::new(None);

/// The route the current connection actually took, set after a successful
/// handshake and reported to the app so the UI can show it:
/// `"direct"`, `"obfuscated"` (dd proxy), or `"faketls:<domain>"` (ee proxy).
static ACTIVE_ROUTE: Mutex<Option<String>> = Mutex::new(None);

fn runtime() -> &'static Runtime {
    RUNTIME.get_or_init(|| Runtime::new().expect("Failed to create Tokio runtime"))
}

fn set_connected(value: bool) {
    CONNECTED.store(value, Ordering::SeqCst);
}

fn is_connected() -> bool {
    CONNECTED.load(Ordering::SeqCst)
}

/// Create the tunnel's TCP socket without connecting it and return its fd.
/// Java must pass this fd to `VpnService.protect()` before calling
/// `connectNative`, so the tunnel traffic bypasses the VPN routes.
fn prepare_socket_impl() -> Result<RawFd> {
    // Discard any previously prepared but never-connected socket.
    let old = PREPARED_SOCKET.swap(-1, Ordering::SeqCst);
    if old >= 0 {
        unsafe { libc::close(old) };
    }
    let fd = unsafe { libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0) };
    if fd < 0 {
        return Err(anyhow!(
            "socket() failed: {}",
            std::io::Error::last_os_error()
        ));
    }
    PREPARED_SOCKET.store(fd, Ordering::SeqCst);
    Ok(fd)
}

#[no_mangle]
pub extern "C" fn Java_io_buzzster_barp_BarpVpnService_prepareSocketNative(
    _env: JNIEnv,
    _class: JClass,
) -> jint {
    android_logger::init_once(
        android_logger::Config::default().with_max_level(log::LevelFilter::Debug),
    );
    match prepare_socket_impl() {
        Ok(fd) => fd as jint,
        Err(e) => {
            log::error!("prepareSocketNative failed: {:?}", e);
            -1
        }
    }
}

/// Read a Java string into an `Option<String>`, treating null/empty as `None`.
fn jstring_to_opt(env: &mut JNIEnv, s: &JString) -> Option<String> {
    if s.is_null() {
        return None;
    }
    match env.get_string(s) {
        Ok(js) => {
            let v: String = js.into();
            if v.trim().is_empty() {
                None
            } else {
                Some(v)
            }
        }
        Err(_) => None,
    }
}

fn set_proxy_impl(env: &mut JNIEnv, addr: JString, secret: JString) {
    let addr = jstring_to_opt(env, &addr);
    let secret = jstring_to_opt(env, &secret);
    match (&addr, &secret) {
        (Some(a), Some(_)) => info!("Proxy configured: {}", a),
        _ => info!("Proxy cleared (direct connection)"),
    }
    set_proxy(addr, secret);
}

#[no_mangle]
pub extern "C" fn Java_io_buzzster_barp_BarpVpnService_setProxyNative(
    mut env: JNIEnv,
    _class: JClass,
    addr: JString,
    secret: JString,
) {
    set_proxy_impl(&mut env, addr, secret);
}

#[no_mangle]
pub extern "C" fn Java_io_buzzster_barp_RustBridge_setProxyNative(
    mut env: JNIEnv,
    _class: JClass,
    addr: JString,
    secret: JString,
) {
    set_proxy_impl(&mut env, addr, secret);
}

fn connect_impl(env: &mut JNIEnv) {
    android_logger::init_once(
        android_logger::Config::default().with_max_level(log::LevelFilter::Debug),
    );

    info!("Rust JNI connectNative called");

    let result: Result<()> = runtime().block_on(async {
        let (session, ipv4, ipv6, write_tx, transport_reader, tcp_reader) =
            connect_and_handshake().await?;
        let mut state = STATE.lock().unwrap();
        state.session = Some(Arc::new(session));
        state.assigned_ipv4 = Some(ipv4);
        state.assigned_ipv6 = Some(ipv6);
        state.write_tx = Some(write_tx);
        state.transport_reader = Some(transport_reader);
        state.tcp_reader = Some(tcp_reader);
        set_connected(true);
        info!("Handshake complete. Assigned IPv4={}, IPv6={}", ipv4, ipv6);
        Ok(())
    });

    if let Err(e) = result {
        log::error!("connectNative failed: {:?}", e);
        set_connected(false);
        let msg = format!("{}", e);
        let _ = env.throw_new("java/lang/RuntimeException", msg);
    }
}

#[no_mangle]
pub extern "C" fn Java_io_buzzster_barp_RustBridge_connectNative(
    mut env: JNIEnv,
    _class: JClass,
) {
    connect_impl(&mut env);
}

#[no_mangle]
pub extern "C" fn Java_io_buzzster_barp_BarpVpnService_connectNative(
    mut env: JNIEnv,
    _class: JClass,
) {
    connect_impl(&mut env);
}

/// Shared implementation of the bridge start. `startBridgeNative` is called from
/// `BarpVpnService` (which owns the tun fd), so the JNI symbol is exported under
/// both class names below.
fn start_bridge_impl(tun_fd: RawFd) {
    info!("Rust JNI startBridgeNative called, tun_fd={}", tun_fd);

    let bundle = {
        let mut state = STATE.lock().unwrap();
        let session = state.session.as_ref().map(Arc::clone);
        let write_tx = state.write_tx.as_ref().cloned();
        let transport_reader = state.transport_reader.take();
        let tcp_reader = state.tcp_reader.take();
        match (session, write_tx, transport_reader, tcp_reader) {
            (Some(s), Some(tx), Some(tr), Some(rd)) => Some((s, tx, tr, rd)),
            _ => None,
        }
    };

    let (session, write_tx, transport_reader, tcp_reader) = match bundle {
        Some(b) => b,
        None => {
            log::error!("startBridgeNative called before connectNative (or bridge already started)");
            return;
        }
    };

    let handle = runtime().spawn(async move {
        if let Err(e) = run_bridge(session, write_tx, transport_reader, tcp_reader, tun_fd).await {
            log::error!("Bridge error: {:?}", e);
        }
        set_connected(false);
    });
    if let Some(old) = BRIDGE_TASK.lock().unwrap().replace(handle) {
        old.abort();
    }
}

fn disconnect_impl() {
    info!("Rust JNI disconnectNative called");
    // Abort the bridge task: dropping it closes the tun fd and the TCP read
    // half, which is what actually tears the tunnel down. Without this the
    // bridge kept running (and the fds leaked) until it hit an I/O error.
    if let Some(handle) = BRIDGE_TASK.lock().unwrap().take() {
        handle.abort();
    }
    let prepared = PREPARED_SOCKET.swap(-1, Ordering::SeqCst);
    if prepared >= 0 {
        unsafe { libc::close(prepared) };
    }
    {
        let mut state = STATE.lock().unwrap();
        *state = GlobalState {
            session: None,
            assigned_ipv4: None,
            assigned_ipv6: None,
            write_tx: None,
            transport_reader: None,
            tcp_reader: None,
        };
    }
    *ACTIVE_ROUTE.lock().unwrap() = None;
    set_connected(false);
}

#[no_mangle]
pub extern "C" fn Java_io_buzzster_barp_RustBridge_startBridgeNative(
    _env: JNIEnv,
    _class: JClass,
    tun_fd: jint,
) {
    start_bridge_impl(tun_fd as RawFd);
}

#[no_mangle]
pub extern "C" fn Java_io_buzzster_barp_BarpVpnService_startBridgeNative(
    _env: JNIEnv,
    _class: JClass,
    tun_fd: jint,
) {
    start_bridge_impl(tun_fd as RawFd);
}

#[no_mangle]
pub extern "C" fn Java_io_buzzster_barp_RustBridge_disconnectNative(_env: JNIEnv, _class: JClass) {
    disconnect_impl();
}

#[no_mangle]
pub extern "C" fn Java_io_buzzster_barp_BarpVpnService_disconnectNative(
    _env: JNIEnv,
    _class: JClass,
) {
    disconnect_impl();
}

#[no_mangle]
pub extern "C" fn Java_io_buzzster_barp_RustBridge_isConnectedNative(
    _env: JNIEnv,
    _class: JClass,
) -> jboolean {
    is_connected() as jboolean
}

fn assigned_ipv4_impl(env: &mut JNIEnv) -> jobject {
    let state = STATE.lock().unwrap();
    match state.assigned_ipv4 {
        Some(ip) => {
            let s = ip.to_string();
            match env.new_string(s) {
                Ok(jstring) => jstring.into_raw(),
                Err(_) => std::ptr::null_mut(),
            }
        }
        None => std::ptr::null_mut(),
    }
}

fn assigned_ipv6_impl(env: &mut JNIEnv) -> jobject {
    let state = STATE.lock().unwrap();
    match state.assigned_ipv6 {
        Some(ip) => {
            let s = ip.to_string();
            match env.new_string(s) {
                Ok(jstring) => jstring.into_raw(),
                Err(_) => std::ptr::null_mut(),
            }
        }
        None => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn Java_io_buzzster_barp_RustBridge_getAssignedIpv4Native(
    mut env: JNIEnv,
    _class: JClass,
) -> jobject {
    assigned_ipv4_impl(&mut env)
}

#[no_mangle]
pub extern "C" fn Java_io_buzzster_barp_RustBridge_getAssignedIpv6Native(
    mut env: JNIEnv,
    _class: JClass,
) -> jobject {
    assigned_ipv6_impl(&mut env)
}

#[no_mangle]
pub extern "C" fn Java_io_buzzster_barp_BarpVpnService_getAssignedIpv4Native(
    mut env: JNIEnv,
    _class: JClass,
) -> jobject {
    assigned_ipv4_impl(&mut env)
}

#[no_mangle]
pub extern "C" fn Java_io_buzzster_barp_BarpVpnService_getAssignedIpv6Native(
    mut env: JNIEnv,
    _class: JClass,
) -> jobject {
    assigned_ipv6_impl(&mut env)
}

/// Returns the route the active connection took (`direct`, `obfuscated`, or
/// `faketls:<domain>`), or null when not connected.
fn proxy_status_impl(env: &mut JNIEnv) -> jobject {
    let route = ACTIVE_ROUTE.lock().unwrap().clone();
    match route {
        Some(s) => match env.new_string(s) {
            Ok(jstring) => jstring.into_raw(),
            Err(_) => std::ptr::null_mut(),
        },
        None => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn Java_io_buzzster_barp_RustBridge_getProxyStatusNative(
    mut env: JNIEnv,
    _class: JClass,
) -> jobject {
    proxy_status_impl(&mut env)
}

#[no_mangle]
pub extern "C" fn Java_io_buzzster_barp_BarpVpnService_getProxyStatusNative(
    mut env: JNIEnv,
    _class: JClass,
) -> jobject {
    proxy_status_impl(&mut env)
}

/// Establish the tunnel's TCP connection. When Java prepared (and protected) a
/// socket via `prepareSocketNative`, connect that exact fd; otherwise fall back
/// to a plain connect (desktop tests, no VpnService involved).
async fn connect_socket() -> Result<TcpStream> {
    let (target_ip, target_port) = tunnel_target()?;
    let fd = PREPARED_SOCKET.swap(-1, Ordering::SeqCst);
    if fd < 0 {
        let addr = format!("{}:{}", target_ip, target_port);
        return Ok(TcpStream::connect(&addr).await?);
    }

    let std_stream = tokio::task::spawn_blocking(move || -> Result<std::net::TcpStream> {
        let addr = libc::sockaddr_in {
            sin_family: libc::AF_INET as libc::sa_family_t,
            sin_port: target_port.to_be(),
            // in_addr is raw network-byte-order bytes; the octets already are.
            sin_addr: libc::in_addr {
                s_addr: u32::from_ne_bytes(target_ip.octets()),
            },
            sin_zero: [0; 8],
        };
        let rc = unsafe {
            libc::connect(
                fd,
                &addr as *const libc::sockaddr_in as *const libc::sockaddr,
                std::mem::size_of::<libc::sockaddr_in>() as libc::socklen_t,
            )
        };
        if rc != 0 {
            let err = std::io::Error::last_os_error();
            unsafe { libc::close(fd) };
            return Err(anyhow!("connect() on protected socket failed: {}", err));
        }
        Ok(unsafe { std::net::TcpStream::from_raw_fd(fd) })
    })
    .await??;

    std_stream.set_nonblocking(true)?;
    Ok(TcpStream::from_std(std_stream)?)
}

/// A boxed async read half, so the direct and FakeTLS-wrapped connections share
/// one type.
type BoxRead = Box<dyn AsyncRead + Unpin + Send>;
type BoxWrite = Box<dyn AsyncWrite + Unpin + Send>;

async fn connect_and_handshake() -> Result<(
    VpnSession,
    Ipv4Addr,
    Ipv6Addr,
    mpsc::Sender<Vec<u8>>,
    TransportReader,
    BoxRead,
)> {
    let socket = connect_socket().await?;
    let (raw_reader, raw_writer) = socket.into_split();

    // Optional MTProto-proxy obfuscation. FakeTLS wraps the stream in a TLS
    // disguise first; both modes then mix the secret into the Obfuscated2 keys.
    let (mut reader, mut writer, secret_key, route_label): (BoxRead, BoxWrite, Option<[u8; 16]>, String) =
        match proxy_secret()? {
            None => {
                info!("Connected directly to {}:{}", SERVER_HOST, SERVER_PORT);
                (Box::new(raw_reader), Box::new(raw_writer), None, "direct".to_string())
            }
            Some(ProxySecret::Simple(key)) => {
                info!("Connected through obfuscated MTProto proxy");
                (Box::new(raw_reader), Box::new(raw_writer), Some(key), "obfuscated".to_string())
            }
            Some(ProxySecret::FakeTls { key, domain }) => {
                let mut raw_reader = raw_reader;
                let mut raw_writer = raw_writer;
                faketls::client_handshake(&mut raw_reader, &mut raw_writer, &key, &domain).await?;
                info!("Connected through FakeTLS MTProto proxy (SNI '{}')", domain);
                (
                    Box::new(FakeTlsReader::new(raw_reader)),
                    Box::new(FakeTlsWriter::new(raw_writer)),
                    Some(key),
                    format!("faketls:{}", domain),
                )
            }
        };

    let (mut transport_reader, mut transport_writer) =
        TransportReader::initiate_obfuscated_with_secret(&mut writer, secret_key).await?;
    info!("Obfuscated transport initiated");

    let (write_tx, mut write_rx) = mpsc::channel::<Vec<u8>>(256);

    let writer_handle = tokio::spawn(async move {
        while let Some(packet) = write_rx.recv().await {
            if let Err(e) = transport_writer.write_packet(&mut writer, &packet).await {
                log::warn!("Client write loop closed: {:?}", e);
                break;
            }
        }
    });

    let (session, ipv4, ipv6) =
        perform_client_handshake(&mut transport_reader, &mut reader, &write_tx).await?;

    // Handshake succeeded: record which route was actually used so the app can
    // display it (direct vs proxy).
    *ACTIVE_ROUTE.lock().unwrap() = Some(route_label);

    // Detach the writer task; it keeps running as long as write_tx (returned and
    // stored) stays alive. The transport reader and TCP read half are returned so
    // the bridge can receive server->client traffic.
    drop(writer_handle);

    Ok((session, ipv4, ipv6, write_tx, transport_reader, reader))
}

/// Aborts a spawned task when dropped. A `tokio::select!` that owns a child
/// JoinHandle does not abort that child if the *outer* task is itself cancelled
/// (e.g. disconnect aborting `run_bridge`): the post-select cleanup never runs.
/// Holding the child's abort handle in a local guard ties the child's lifetime
/// to the outer task's stack, so cancellation tears the child down too. Without
/// this, the TUN reader task kept a clone of the tun fd alive and the Android
/// VPN interface never went down after disconnect.
struct AbortOnDrop(tokio::task::AbortHandle);
impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

async fn run_bridge(
    session: Arc<VpnSession>,
    write_tx: mpsc::Sender<Vec<u8>>,
    mut transport_reader: TransportReader,
    mut tcp_reader: BoxRead,
    tun_fd: RawFd,
) -> Result<()> {
    // A TUN device is a non-blocking character device, not a regular file, so
    // wrap the fd in tokio's AsyncFd for readiness-based (epoll) async I/O.
    // tokio::fs::File would surface EAGAIN as a hard error instead of waiting.
    let tun_file = unsafe { std::fs::File::from_raw_fd(tun_fd) };
    set_nonblocking(tun_file.as_raw_fd())?;
    let tun = Arc::new(AsyncFd::new(tun_file)?);

    // TUN -> server: read raw IP packets from the tun device, wrap them as
    // vpn.packet and send them encrypted to the server.
    let write_tx_clone = write_tx.clone();
    let session_clone = Arc::clone(&session);
    let tun_read = Arc::clone(&tun);
    let mut tun_to_server = tokio::spawn(async move {
        let mut buf = vec![0u8; 8192];
        loop {
            let n = match read_tun(&tun_read, &mut buf).await {
                Ok(0) => break,
                Ok(n) => n,
                Err(e) => {
                    log::warn!("TUN read error: {:?}", e);
                    break;
                }
            };
            let body = rpc::serialize_vpn_packet(&buf[..n]);
            if let Err(e) = send_encrypted(&body, &session_clone, true, &write_tx_clone).await {
                log::warn!("Failed to send packet to server: {:?}", e);
                break;
            }
        }
    });

    // Abort the TUN reader if this task is cancelled (disconnect) so its clone of
    // the tun fd is released and the OS tears the VPN interface down.
    let _tun_to_server_guard = AbortOnDrop(tun_to_server.abort_handle());

    // server -> TUN: read encrypted messages from the server, decrypt them and
    // write the contained IP packets to the tun device. Non-vpn.packet service
    // messages (e.g. new_session_created) are decrypted and ignored.
    let tun_write = Arc::clone(&tun);
    let server_to_tun = async {
        loop {
            let payload = transport_reader.read_packet(&mut tcp_reader).await?;
            let ip_packet = match decrypt_server_message(&payload, &session)? {
                Some(pkt) => pkt,
                None => continue,
            };
            write_tun(&tun_write, &ip_packet).await?;
        }
    };

    tokio::select! {
        _ = &mut tun_to_server => {
            log::info!("TUN reader loop ended; closing bridge");
        }
        res = server_to_tun => {
            let res: Result<()> = res;
            if let Err(e) = res {
                log::warn!("Server reader loop ended: {:?}", e);
            }
        }
    }

    tun_to_server.abort();
    drop(write_tx);
    Ok(())
}

fn set_nonblocking(fd: RawFd) -> Result<()> {
    unsafe {
        let flags = libc::fcntl(fd, libc::F_GETFL, 0);
        if flags < 0 {
            return Err(anyhow!("fcntl(F_GETFL) failed"));
        }
        if libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) < 0 {
            return Err(anyhow!("fcntl(F_SETFL, O_NONBLOCK) failed"));
        }
    }
    Ok(())
}

/// Read one packet from the TUN device, awaiting readiness on EAGAIN.
async fn read_tun(tun: &AsyncFd<std::fs::File>, buf: &mut [u8]) -> std::io::Result<usize> {
    loop {
        let mut guard = tun.readable().await?;
        match guard.try_io(|inner| {
            use std::io::Read;
            let mut f: &std::fs::File = inner.get_ref();
            f.read(buf)
        }) {
            Ok(result) => return result,
            Err(_would_block) => continue,
        }
    }
}

/// Write one packet to the TUN device, awaiting writability on EAGAIN.
async fn write_tun(tun: &AsyncFd<std::fs::File>, data: &[u8]) -> std::io::Result<()> {
    loop {
        let mut guard = tun.writable().await?;
        match guard.try_io(|inner| {
            use std::io::Write;
            let mut f: &std::fs::File = inner.get_ref();
            f.write(data).map(|_| ())
        }) {
            Ok(result) => return result,
            Err(_would_block) => continue,
        }
    }
}

/// Decrypt one server->client MTProto message. Returns the inner IP packet when
/// the body is a vpn.packet, or `None` for service messages we ignore.
fn decrypt_server_message(payload: &[u8], session: &VpnSession) -> Result<Option<Vec<u8>>> {
    if payload.len() < 24 {
        return Err(anyhow!("Server packet too short"));
    }
    let incoming_key_id = i64::from_le_bytes(payload[0..8].try_into().unwrap());
    if incoming_key_id != session.auth_key_id {
        return Err(anyhow!("Server packet auth_key_id mismatch"));
    }
    let mut msg_key = [0u8; 16];
    msg_key.copy_from_slice(&payload[8..24]);
    let encrypted_len = (payload.len() - 24) & !15;
    // Message sent by the server: direction constant x = 8.
    let decrypted = crypto_ige::decrypt_message_x(
        &session.auth_key,
        &msg_key,
        &payload[24..24 + encrypted_len],
        8,
    )?;
    if decrypted.len() < 32 {
        return Err(anyhow!("Decrypted server payload too short"));
    }
    if &decrypted[8..16] != &session.session_id {
        return Err(anyhow!("Server session ID mismatch"));
    }
    let msg_len = i32::from_le_bytes(decrypted[28..32].try_into().unwrap()) as usize;
    if decrypted.len() < 32 + msg_len || msg_len < 4 {
        return Err(anyhow!("Decrypted server payload size mismatch"));
    }
    let body = &decrypted[32..32 + msg_len];
    let constructor_id = u32::from_le_bytes(body[0..4].try_into().unwrap());
    if constructor_id == rpc::VPN_PACKET_CONSTRUCTOR_ID {
        Ok(Some(rpc::parse_vpn_packet(body)?))
    } else {
        // e.g. new_session_created / msgs_ack — nothing to forward to the TUN.
        Ok(None)
    }
}

async fn perform_client_handshake<R: AsyncRead + Unpin>(
    transport_reader: &mut TransportReader,
    reader: &mut R,
    write_tx: &mpsc::Sender<Vec<u8>>,
) -> Result<(VpnSession, Ipv4Addr, Ipv6Addr)> {
    // 1. req_pq_multi
    let mut nonce = [0u8; 16];
    rand::thread_rng().fill(&mut nonce);
    let req_pq = functions::ReqPqMulti { nonce };
    write_plaintext(&req_pq, write_tx).await?;

    // 2. resPQ
    let payload = transport_reader.read_packet(reader).await?;
    let (msg_data, _msg_id) = parse_plaintext(&payload)?;
    let res_pq = enums::ResPq::deserialize(&mut Cursor::from_slice(&msg_data))
        .map_err(|e| anyhow!("Failed to deserialize resPQ: {}", e))?;
    let res_pq = match res_pq {
        enums::ResPq::Pq(r) => r,
    };
    if res_pq.nonce != nonce {
        return Err(anyhow!("Nonce mismatch in resPQ"));
    }
    let server_nonce = res_pq.server_nonce;
    let rsa_fingerprint = *res_pq
        .server_public_key_fingerprints
        .first()
        .ok_or_else(|| anyhow!("No server public key fingerprint"))? as u64;
    info!("Received resPQ, fingerprint {:#x}", rsa_fingerprint);

    let public_key = load_public_key_by_fingerprint(rsa_fingerprint)?;

    // 3. req_DH_params
    let mut new_nonce = [0u8; 32];
    rand::thread_rng().fill(&mut new_nonce);

    let pq_val = 2342281u64;
    let p_val = 1721u64;
    let q_val = 1361u64;
    let pq_bytes = big_endian_minimal(pq_val);
    let p_bytes = big_endian_minimal(p_val);
    let q_bytes = big_endian_minimal(q_val);

    let inner_data = enums::PQInnerData::Data(types::PQInnerData {
        pq: pq_bytes.clone(),
        p: p_bytes.clone(),
        q: q_bytes.clone(),
        nonce,
        server_nonce,
        new_nonce,
    });

    let encrypted_data = rsa_pad_encrypt(&inner_data, &public_key)?;
    let mut req_dh_body = Vec::new();
    req_dh_body.extend_from_slice(&functions::ReqDhParams::CONSTRUCTOR_ID.to_le_bytes());
    req_dh_body.extend_from_slice(&nonce);
    req_dh_body.extend_from_slice(&server_nonce);
    serialize_bytes_tl(&mut req_dh_body, &p_bytes);
    serialize_bytes_tl(&mut req_dh_body, &q_bytes);
    req_dh_body.extend_from_slice(&(rsa_fingerprint as i64).to_le_bytes());
    serialize_bytes_tl(&mut req_dh_body, &encrypted_data);
    write_plaintext_raw(&req_dh_body, write_tx).await?;
    info!("Sent req_DH_params");

    // 4. server_DH_params_ok
    let payload = transport_reader.read_packet(reader).await?;
    let (msg_data, _msg_id) = parse_plaintext(&payload)?;
    let dh_params = enums::ServerDhParams::deserialize(&mut Cursor::from_slice(&msg_data))
        .map_err(|e| anyhow!("Failed to deserialize server_DH_params: {}", e))?;
    let dh_ok = match dh_params {
        enums::ServerDhParams::Ok(o) => o,
        _ => return Err(anyhow!("server_DH_params not ok")),
    };
    if dh_ok.nonce != nonce || dh_ok.server_nonce != server_nonce {
        return Err(anyhow!("Nonce mismatch in server_DH_params"));
    }

    let (tmp_key, tmp_iv) = derive_temp_key_iv(&server_nonce, &new_nonce);
    let mut decrypted = dh_ok.encrypted_answer.clone();
    crypto_ige::decrypt_ige_raw(&tmp_key, &tmp_iv, &mut decrypted)?;
    if decrypted.len() < 20 {
        return Err(anyhow!("ServerDhInnerData too short"));
    }
    let expected_hash = &decrypted[0..20];
    let mut inner_cursor = Cursor::from_slice(&decrypted[20..]);
    let inner_dh = enums::ServerDhInnerData::deserialize(&mut inner_cursor)
        .map_err(|e| anyhow!("Failed to deserialize ServerDhInnerData: {}", e))?;
    let consumed = inner_cursor.pos();
    let inner_payload = &decrypted[20..20 + consumed];
    let mut h = Sha1::new();
    h.update(inner_payload);
    if h.finalize().as_slice() != expected_hash {
        return Err(anyhow!("ServerDhInnerData SHA-1 integrity check failed"));
    }
    let dh_data = match inner_dh {
        enums::ServerDhInnerData::Data(d) => d,
    };
    let p_dh = BigUint::from_bytes_be(&dh_data.dh_prime);
    let g_dh = BigUint::from(dh_data.g as u64);
    let g_a = BigUint::from_bytes_be(&dh_data.g_a);

    validate_g_a(&g_a, &p_dh)?;

    // 5. set_client_DH_params
    let mut b_bytes = vec![0u8; 256];
    rand::thread_rng().fill(&mut b_bytes[..]);
    let b = BigUint::from_bytes_be(&b_bytes);
    let g_b = g_dh.modpow(&b, &p_dh);

    validate_g_a(&g_b, &p_dh)?;

    let client_inner = enums::ClientDhInnerData::Data(types::ClientDhInnerData {
        nonce,
        server_nonce,
        retry_id: 0,
        g_b: g_b.to_bytes_be(),
    });
    let mut serialized = Vec::new();
    client_inner.serialize(&mut serialized);
    let mut h = Sha1::new();
    h.update(&serialized);
    let hash = h.finalize();
    let mut to_encrypt = Vec::new();
    to_encrypt.extend_from_slice(&hash);
    to_encrypt.extend_from_slice(&serialized);
    let pad_len = (16 - (to_encrypt.len() % 16)) % 16;
    let mut pad = vec![0u8; pad_len];
    rand::thread_rng().fill(&mut pad[..]);
    to_encrypt.extend_from_slice(&pad);
    crypto_ige::encrypt_ige_raw(&tmp_key, &tmp_iv, &mut to_encrypt)?;

    let mut set_dh_body = Vec::new();
    set_dh_body.extend_from_slice(&functions::SetClientDhParams::CONSTRUCTOR_ID.to_le_bytes());
    set_dh_body.extend_from_slice(&nonce);
    set_dh_body.extend_from_slice(&server_nonce);
    serialize_bytes_tl(&mut set_dh_body, &to_encrypt);
    write_plaintext_raw(&set_dh_body, write_tx).await?;
    info!("Sent set_client_DH_params");

    // 6. dh_gen_ok
    let payload = transport_reader.read_packet(reader).await?;
    let (msg_data, _msg_id) = parse_plaintext(&payload)?;
    let answer = enums::SetClientDhParamsAnswer::deserialize(&mut Cursor::from_slice(&msg_data))
        .map_err(|e| anyhow!("Failed to deserialize dh_gen answer: {}", e))?;
    let dh_gen = match answer {
        enums::SetClientDhParamsAnswer::DhGenOk(ok) => ok,
        other => return Err(anyhow!("DH handshake failed: {:?}", other)),
    };
    if dh_gen.nonce != nonce || dh_gen.server_nonce != server_nonce {
        return Err(anyhow!("Nonce mismatch in dh_gen_ok"));
    }
    info!("DH handshake successful");

    let auth_key_int = g_a.modpow(&b, &p_dh);
    let mut auth_key_bytes = auth_key_int.to_bytes_be();
    if auth_key_bytes.len() < 256 {
        let mut padded = vec![0u8; 256 - auth_key_bytes.len()];
        padded.extend_from_slice(&auth_key_bytes);
        auth_key_bytes = padded;
    }
    let mut auth_key = [0u8; 256];
    auth_key.copy_from_slice(&auth_key_bytes[..256]);

    let auth_key_id = {
        let mut hasher = Sha1::new();
        hasher.update(&auth_key);
        let hash = hasher.finalize();
        let mut id_bytes = [0u8; 8];
        id_bytes.copy_from_slice(&hash[12..20]);
        i64::from_le_bytes(id_bytes)
    };

    // Confirm the server derived the same auth key: new_nonce_hash1 binds the
    // shared key, so a mismatch means DH disagreement — fail now instead of
    // later on every encrypted packet.
    let expected_hash1 = compute_new_nonce_hash(&new_nonce, &auth_key, 1);
    if dh_gen.new_nonce_hash1 != expected_hash1 {
        return Err(anyhow!("dh_gen_ok new_nonce_hash1 mismatch (auth key disagreement)"));
    }
    let salt = {
        let mut s = [0u8; 8];
        for i in 0..8 {
            s[i] = new_nonce[i] ^ server_nonce[i];
        }
        i64::from_le_bytes(s)
    };
    // Session id is derived deterministically from the handshake nonces so it
    // matches the server without being transmitted (see server compute_session_id).
    let session_id = compute_session_id(&new_nonce, &server_nonce);

    // 7. First encrypted message: vpn.config
    let payload_enc = transport_reader.read_packet(reader).await?;
    let (ipv4, ipv6) = receive_vpn_config(&payload_enc, &auth_key, auth_key_id, &session_id).await?;
    info!("Received vpn.config: IPv4={}, IPv6={}", ipv4, ipv6);

    let session = VpnSession {
        auth_key,
        auth_key_id,
        salt,
        session_id,
        server_seq_no: Mutex::new(1),
    };

    Ok((session, ipv4, ipv6))
}

async fn receive_vpn_config(
    payload: &[u8],
    auth_key: &[u8; 256],
    auth_key_id: i64,
    session_id: &[u8; 8],
) -> Result<(Ipv4Addr, Ipv6Addr)> {
    if payload.len() < 24 {
        return Err(anyhow!("Config packet too short"));
    }
    let incoming_key_id = i64::from_le_bytes(payload[0..8].try_into().unwrap());
    if incoming_key_id != auth_key_id {
        return Err(anyhow!("Config packet auth_key_id mismatch"));
    }
    let mut msg_key = [0u8; 16];
    msg_key.copy_from_slice(&payload[8..24]);
    let encrypted_len = (payload.len() - 24) & !15;
    let encrypted_data = &payload[24..24 + encrypted_len];
    // Message sent by the server: direction constant x = 8.
    let decrypted = crypto_ige::decrypt_message_x(auth_key, &msg_key, encrypted_data, 8)?;
    if decrypted.len() < 32 {
        return Err(anyhow!("Decrypted config too short"));
    }
    if &decrypted[8..16] != session_id {
        return Err(anyhow!("Config session ID mismatch"));
    }
    let msg_len = i32::from_le_bytes(decrypted[28..32].try_into().unwrap()) as usize;
    let body = &decrypted[32..32 + msg_len];
    let (ipv4_bytes, ipv6_bytes) = rpc::parse_vpn_config(body)?;
    Ok((Ipv4Addr::from(ipv4_bytes), Ipv6Addr::from(ipv6_bytes)))
}

fn rsa_pad_encrypt(
    inner: &enums::PQInnerData,
    public_key: &rsa::RsaPublicKey,
) -> Result<Vec<u8>> {
    let mut data = Vec::new();
    inner.serialize(&mut data);
    if data.len() > 144 {
        return Err(anyhow!("PQInnerData too long for RSA_PAD"));
    }

    let mut data_with_padding = data.clone();
    let pad_len = 192 - data.len();
    if pad_len > 0 {
        let mut pad = vec![0u8; pad_len];
        rand::thread_rng().fill(&mut pad[..]);
        data_with_padding.extend_from_slice(&pad);
    }

    let data_pad_reversed: Vec<u8> = data_with_padding.iter().copied().rev().collect();

    // MTProto RSA_PAD uses *raw* RSA (m^e mod n) over the 256-byte block, which
    // the server inverts with c^d mod n. PKCS#1 padding must NOT be applied, and
    // it could not be anyway (a 256-byte message does not fit PKCS#1 v1.5 for a
    // 2048-bit key). We retry with a fresh temp_key until the block is a valid
    // residue (< n).
    let n_num = BigUint::from_bytes_be(&public_key.n().to_bytes_be());
    let e_num = BigUint::from_bytes_be(&public_key.e().to_bytes_be());

    loop {
        let mut temp_key = [0u8; 32];
        rand::thread_rng().fill(&mut temp_key);

        let mut to_hash = Vec::new();
        to_hash.extend_from_slice(&temp_key);
        to_hash.extend_from_slice(&data_with_padding);
        let hash = {
            let mut h = Sha256Hash::new();
            h.update(&to_hash);
            h.finalize()
        };

        let mut data_with_hash = Vec::new();
        data_with_hash.extend_from_slice(&data_pad_reversed);
        data_with_hash.extend_from_slice(&hash);

        crypto_ige::encrypt_ige_raw(&temp_key, &[0u8; 32], &mut data_with_hash)?;

        let aes_hash = {
            let mut h = Sha256Hash::new();
            h.update(&data_with_hash);
            h.finalize()
        };

        let temp_key_xor: Vec<u8> = temp_key
            .iter()
            .zip(aes_hash.iter())
            .map(|(a, b)| a ^ b)
            .collect();

        let mut key_aes_encrypted = Vec::with_capacity(256);
        key_aes_encrypted.extend_from_slice(&temp_key_xor);
        key_aes_encrypted.extend_from_slice(&data_with_hash);

        let m = BigUint::from_bytes_be(&key_aes_encrypted);
        if m >= n_num {
            continue;
        }
        let c = m.modpow(&e_num, &n_num);
        let mut encrypted = c.to_bytes_be();
        if encrypted.len() < 256 {
            let mut padded = vec![0u8; 256 - encrypted.len()];
            padded.extend_from_slice(&encrypted);
            encrypted = padded;
        }
        return Ok(encrypted);
    }
}

fn load_public_key_by_fingerprint(fingerprint: u64) -> Result<rsa::RsaPublicKey> {
    // Embed the server public key PEM at build time, exactly like Telegram
    // embeds its hard-coded server public keys in tgnet.
    const SERVER_PUBLIC_KEY_PEM: &str = "-----BEGIN RSA PUBLIC KEY-----\n\
MIIBCgKCAQEA1sEPmXKgjZ++EokQ3ru2C8Jl2qD8BLOsIqNfvTg/B6RjX9eJuQr9\n\
/JCmYFVW/rB9Zit2cogiC8NOLX3E4x+ml7/XdLRAJscEvTYfa6U5AREhJ0z1vxgG\n\
n/FvKvBt3ODXwta6aeTJAtCgyCzNpl2uFEPprALlpFUlXVyVtEWZkgo2iex76DTT\n\
L1L0uRo0ZtXE9cgacfR2a3jDqNnF55a0nmbxkbzq6DQaW30Pde7Rs7WobgDM8DBK\n\
+dNjo/e1CBpKgIZOSmMh8jJWQFWh/NSntN7NRlCy63+it87wl8q0cfJCYEeJBoeX\n\
yiIzl0RFlGOjUUaePVjTV6/sinCAGPFUDwIDAQAB\n\
-----END RSA PUBLIC KEY-----";
    const SERVER_FINGERPRINT: u64 = 0x48544974114217e5;

    if fingerprint != SERVER_FINGERPRINT {
        return Err(anyhow!(
            "Fingerprint mismatch: expected {:#x}, got {:#x}",
            SERVER_FINGERPRINT,
            fingerprint
        ));
    }

    let public_key = rsa::RsaPublicKey::from_pkcs1_pem(SERVER_PUBLIC_KEY_PEM)?;
    let computed = crypto_dh::get_fingerprint(&public_key);
    if computed != SERVER_FINGERPRINT {
        return Err(anyhow!(
            "Computed fingerprint mismatch: expected {:#x}, got {:#x}",
            SERVER_FINGERPRINT,
            computed
        ));
    }
    Ok(public_key)
}

fn validate_g_a(g_a: &BigUint, p: &BigUint) -> Result<()> {
    if g_a >= p || g_a == &BigUint::from(1u32) {
        return Err(anyhow!("g_a out of range"));
    }
    let g_a_bytes = g_a.to_bytes_be();
    if g_a_bytes.len() > 256 {
        return Err(anyhow!("g_a too large"));
    }
    let min_bits = 2048 - 64;
    if g_a.bits() < min_bits {
        return Err(anyhow!("g_a too small"));
    }
    let p_minus_ga = p - g_a;
    if p_minus_ga.bits() < min_bits {
        return Err(anyhow!("g_a too close to p"));
    }
    Ok(())
}

fn big_endian_minimal(value: u64) -> Vec<u8> {
    let be = value.to_be_bytes();
    let start = be.iter().position(|&b| b != 0).unwrap_or(0);
    be[start..].to_vec()
}

async fn write_plaintext<T: Serializable>(msg: &T, write_tx: &mpsc::Sender<Vec<u8>>) -> Result<()> {
    let mut payload = Vec::new();
    msg.serialize(&mut payload);
    write_plaintext_raw(&payload, write_tx).await
}

async fn write_plaintext_raw(payload: &[u8], write_tx: &mpsc::Sender<Vec<u8>>) -> Result<()> {
    let mut packet = Vec::new();
    packet.extend_from_slice(&[0u8; 8]);
    let msg_id = generate_plaintext_msg_id();
    packet.extend_from_slice(&msg_id.to_le_bytes());
    packet.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    packet.extend_from_slice(payload);

    write_tx
        .send(packet)
        .await
        .map_err(|_| anyhow!("Writer channel closed"))
}

fn serialize_bytes_tl(buf: &mut Vec<u8>, bytes: &[u8]) {
    let len = bytes.len();
    if len < 254 {
        buf.push(len as u8);
    } else {
        buf.push(254);
        buf.extend_from_slice(&(len as u32).to_le_bytes()[0..3]);
    }
    buf.extend_from_slice(bytes);
    let padding = (4 - (buf.len() % 4)) % 4;
    buf.extend_from_slice(&vec![0u8; padding]);
}

fn parse_plaintext(payload: &[u8]) -> Result<(Vec<u8>, i64)> {
    if payload.len() < 20 {
        return Err(anyhow!("Plaintext packet too short"));
    }
    let msg_id = i64::from_le_bytes(payload[8..16].try_into().unwrap());
    let msg_len = u32::from_le_bytes(payload[16..20].try_into().unwrap()) as usize;
    if payload.len() < 20 + msg_len {
        return Err(anyhow!("Plaintext payload length mismatch"));
    }
    Ok((payload[20..20 + msg_len].to_vec(), msg_id))
}

fn derive_temp_key_iv(server_nonce: &[u8; 16], new_nonce: &[u8; 32]) -> ([u8; 32], [u8; 32]) {
    let mut sha_a = Sha1::new();
    sha_a.update(new_nonce);
    sha_a.update(server_nonce);
    let sha1_a = sha_a.finalize();

    let mut sha_b = Sha1::new();
    sha_b.update(server_nonce);
    sha_b.update(new_nonce);
    let sha1_b = sha_b.finalize();

    let mut sha_c = Sha1::new();
    sha_c.update(new_nonce);
    sha_c.update(new_nonce);
    let sha1_c = sha_c.finalize();

    let mut key = [0u8; 32];
    key[0..20].copy_from_slice(&sha1_a);
    key[20..32].copy_from_slice(&sha1_b[0..12]);

    let mut iv = [0u8; 32];
    iv[0..8].copy_from_slice(&sha1_b[12..20]);
    iv[8..28].copy_from_slice(&sha1_c);
    iv[28..32].copy_from_slice(&new_nonce[0..4]);

    (key, iv)
}

static LAST_PLAINTEXT_MSG_ID: AtomicI64 = AtomicI64::new(0);
static LAST_RESPONSE_MSG_ID: AtomicI64 = AtomicI64::new(0);

/// Strictly-increasing MTProto message id against `last`. High 32 bits carry the
/// unix time in seconds and the low 32 bits the fractional second (scaled to
/// 2^32), so ids issued within the same second stay distinct and ordered — the
/// server rejects any client msg_id that is not greater than the previous one.
fn next_msg_id(last: &AtomicI64) -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    let secs = now.as_secs() as i64;
    let nanos = now.subsec_nanos() as u64;
    let frac = ((nanos << 32) / 1_000_000_000) as i64 & 0xffff_fffc;
    let candidate = (secs << 32) | frac | 1;
    loop {
        let prev = last.load(Ordering::SeqCst);
        let target = if candidate > prev { candidate } else { (prev & !3) + 5 };
        if last
            .compare_exchange(prev, target, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            return target;
        }
    }
}

/// Deterministically derive the 8-byte session id shared by both peers from the
/// handshake nonces. Must match the server's `compute_session_id` exactly.
fn compute_session_id(new_nonce: &[u8; 32], server_nonce: &[u8; 16]) -> [u8; 8] {
    let mut h = Sha256Hash::new();
    h.update(new_nonce);
    h.update(server_nonce);
    let digest = h.finalize();
    let mut id = [0u8; 8];
    id.copy_from_slice(&digest[0..8]);
    id
}

fn compute_new_nonce_hash(new_nonce: &[u8; 32], auth_key: &[u8; 256], hash_num: u8) -> [u8; 16] {
    let mut sha_auth = Sha1::new();
    sha_auth.update(auth_key);
    let auth_hash = sha_auth.finalize();
    let auth_key_aux_hash = &auth_hash[0..8];

    let mut sha_final = Sha1::new();
    sha_final.update(new_nonce);
    sha_final.update(&[hash_num]);
    sha_final.update(auth_key_aux_hash);
    let final_hash = sha_final.finalize();

    let mut result = [0u8; 16];
    result.copy_from_slice(&final_hash[4..20]);
    result
}

fn generate_plaintext_msg_id() -> i64 {
    next_msg_id(&LAST_PLAINTEXT_MSG_ID)
}

async fn send_encrypted(
    body: &[u8],
    session: &VpnSession,
    content_related: bool,
    write_tx: &mpsc::Sender<Vec<u8>>,
) -> Result<()> {
    let salt_bytes = session.salt.to_le_bytes();
    let mut plaintext = Vec::new();
    plaintext.extend_from_slice(&salt_bytes);
    plaintext.extend_from_slice(&session.session_id);
    plaintext.extend_from_slice(&generate_response_msg_id().to_le_bytes());
    let seq = if content_related {
        let mut lock = session.server_seq_no.lock().unwrap();
        let v = *lock;
        *lock = v + 2;
        v
    } else {
        let v = *session.server_seq_no.lock().unwrap();
        (v - 1).max(0)
    };
    plaintext.extend_from_slice(&seq.to_le_bytes());
    plaintext.extend_from_slice(&(body.len() as u32).to_le_bytes());
    plaintext.extend_from_slice(body);

    // Client is the sender: direction constant x = 0.
    let (encrypted, msg_key) = crypto_ige::encrypt_message_x(&session.auth_key, &plaintext, 0)?;
    let mut packet = Vec::new();
    packet.extend_from_slice(&session.auth_key_id.to_le_bytes());
    packet.extend_from_slice(&msg_key);
    packet.extend_from_slice(&encrypted);
    write_tx
        .send(packet)
        .await
        .map_err(|_| anyhow!("Writer channel closed"))
}

fn generate_response_msg_id() -> i64 {
    next_msg_id(&LAST_RESPONSE_MSG_ID)
}
