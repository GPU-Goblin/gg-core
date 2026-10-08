//! NAT-PMP client (RFC 6886) used to obtain a port mapping from the local gateway (ADR-0013).
//!
//! This is the part of NAT traversal that can be implemented and verified honestly: many home and
//! laboratory gateways answer NAT-PMP on UDP 5351. When the gateway answers, the worker announces
//! the external address the gateway gave it. When it does not, the worker falls back to the address
//! the control plane observed, and the control plane's dial-back decides whether that address is
//! advertised at all. Hole punching between two NATed peers is **not** implemented: it needs a UDP
//! transport and a rendezvous protocol (documented in ADR-0013).

use std::net::{IpAddr, Ipv4Addr, SocketAddr, SocketAddrV4};
use std::time::Duration;

/// Port the protocol reserves on the gateway.
pub const NAT_PMP_PORT: u16 = 5351;
/// Protocol version byte.
const VERSION: u8 = 0;
const OP_EXTERNAL_ADDRESS: u8 = 0;
const OP_MAP_TCP: u8 = 2;
/// Responses set the high bit of the opcode.
const RESPONSE_BIT: u8 = 128;

/// Result of a successful mapping request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mapping {
    /// Port the gateway mapped on its external side.
    pub external_port: u16,
    /// Internal port that was mapped.
    pub internal_port: u16,
    /// Lifetime granted by the gateway, in seconds.
    pub lifetime_secs: u32,
}

/// Errors that keep peer sharing working (the caller falls back to the observed address).
#[derive(Debug, thiserror::Error)]
pub enum NatPmpError {
    /// The gateway did not answer in time.
    #[error("no answer from the gateway at {0}")]
    NoAnswer(SocketAddr),
    /// The gateway answered with a protocol error.
    #[error("gateway refused the request (result code {0})")]
    Refused(u16),
    /// The answer was not a valid NAT-PMP response to our request.
    #[error("malformed answer: {0}")]
    Malformed(String),
    /// Local I/O.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// No default gateway could be determined.
    #[error("no IPv4 default gateway found")]
    NoGateway,
}

/// Default IPv4 gateway read from the kernel routing table (Linux).
pub fn default_gateway() -> Result<Ipv4Addr, NatPmpError> {
    let text = std::fs::read_to_string("/proc/net/route")?;
    parse_default_gateway(&text).ok_or(NatPmpError::NoGateway)
}

/// Parse `/proc/net/route` and return the first IPv4 default gateway.
pub fn parse_default_gateway(text: &str) -> Option<Ipv4Addr> {
    for line in text.lines().skip(1) {
        let mut cols = line.split_whitespace();
        let _iface = cols.next()?;
        let dest = cols.next()?;
        let gateway = cols.next()?;
        if dest != "00000000" {
            continue;
        }
        let raw = u32::from_str_radix(gateway, 16).ok()?;
        // /proc/net/route stores addresses little-endian.
        let addr = Ipv4Addr::from(raw.to_le_bytes());
        if !addr.is_unspecified() {
            return Some(addr);
        }
    }
    None
}

fn map_request(internal_port: u16, suggested_external: u16, lifetime_secs: u32) -> [u8; 12] {
    let mut req = [0u8; 12];
    req[0] = VERSION;
    req[1] = OP_MAP_TCP;
    // req[2..4] reserved, zero
    req[4..6].copy_from_slice(&internal_port.to_be_bytes());
    req[6..8].copy_from_slice(&suggested_external.to_be_bytes());
    req[8..12].copy_from_slice(&lifetime_secs.to_be_bytes());
    req
}

/// Parse a mapping response (16 bytes) for `internal_port`.
pub fn parse_map_response(buf: &[u8], internal_port: u16) -> Result<Mapping, NatPmpError> {
    if buf.len() < 16 {
        return Err(NatPmpError::Malformed(format!(
            "{} bytes, expected 16",
            buf.len()
        )));
    }
    if buf[0] != VERSION {
        return Err(NatPmpError::Malformed(format!("version {}", buf[0])));
    }
    if buf[1] != OP_MAP_TCP + RESPONSE_BIT {
        return Err(NatPmpError::Malformed(format!("opcode {}", buf[1])));
    }
    let result = u16::from_be_bytes([buf[2], buf[3]]);
    if result != 0 {
        return Err(NatPmpError::Refused(result));
    }
    let answered_internal = u16::from_be_bytes([buf[8], buf[9]]);
    if answered_internal != internal_port {
        return Err(NatPmpError::Malformed(format!(
            "answer maps internal port {answered_internal}, asked {internal_port}"
        )));
    }
    Ok(Mapping {
        external_port: u16::from_be_bytes([buf[10], buf[11]]),
        internal_port: answered_internal,
        lifetime_secs: u32::from_be_bytes([buf[12], buf[13], buf[14], buf[15]]),
    })
}

/// Parse an external-address response (12 bytes).
pub fn parse_external_address(buf: &[u8]) -> Result<Ipv4Addr, NatPmpError> {
    if buf.len() < 12 {
        return Err(NatPmpError::Malformed(format!(
            "{} bytes, expected 12",
            buf.len()
        )));
    }
    if buf[0] != VERSION || buf[1] != OP_EXTERNAL_ADDRESS + RESPONSE_BIT {
        return Err(NatPmpError::Malformed("not an address response".into()));
    }
    let result = u16::from_be_bytes([buf[2], buf[3]]);
    if result != 0 {
        return Err(NatPmpError::Refused(result));
    }
    Ok(Ipv4Addr::new(buf[8], buf[9], buf[10], buf[11]))
}

async fn exchange(
    gateway: SocketAddr,
    request: &[u8],
    timeout: Duration,
    tries: u32,
) -> Result<Vec<u8>, NatPmpError> {
    let socket = tokio::net::UdpSocket::bind(SocketAddr::from((Ipv4Addr::UNSPECIFIED, 0))).await?;
    socket.connect(gateway).await?;
    let mut buf = vec![0u8; 32];
    for _ in 0..tries.max(1) {
        socket.send(request).await?;
        match tokio::time::timeout(timeout, socket.recv(&mut buf)).await {
            Ok(Ok(n)) => {
                buf.truncate(n);
                return Ok(buf);
            }
            Ok(Err(e)) => return Err(e.into()),
            Err(_) => continue, // RFC 6886 retries with growing timeouts; two tries is enough here
        }
    }
    Err(NatPmpError::NoAnswer(gateway))
}

/// Ask `gateway` to map `internal_port` to an external TCP port for `lifetime_secs`.
pub async fn request_mapping(
    gateway: SocketAddr,
    internal_port: u16,
    lifetime_secs: u32,
    timeout: Duration,
) -> Result<Mapping, NatPmpError> {
    let request = map_request(internal_port, internal_port, lifetime_secs);
    let answer = exchange(gateway, &request, timeout, 2).await?;
    parse_map_response(&answer, internal_port)
}

/// Delete a mapping: RFC 6886 asks for a lifetime of 0 and an external port of 0.
pub async fn delete_mapping(
    gateway: SocketAddr,
    internal_port: u16,
    timeout: Duration,
) -> Result<(), NatPmpError> {
    let request = map_request(internal_port, 0, 0);
    let answer = exchange(gateway, &request, timeout, 2).await?;
    // The answer repeats the internal port; the external port is meaningless for a deletion.
    parse_map_response(&answer, internal_port).map(|_| ())
}

/// Ask `gateway` for the address it presents to the outside.
pub async fn request_external_address(
    gateway: SocketAddr,
    timeout: Duration,
) -> Result<Ipv4Addr, NatPmpError> {
    let answer = exchange(gateway, &[VERSION, OP_EXTERNAL_ADDRESS], timeout, 2).await?;
    parse_external_address(&answer)
}

/// Full attempt: find the gateway, ask for its external address and a mapping of `internal_port`.
/// Returns the address to announce. Any failure is reported to the caller, which falls back to the
/// address observed by the control plane.
pub async fn map_port(
    internal_port: u16,
    lifetime_secs: u32,
    timeout: Duration,
    gateway_override: Option<SocketAddr>,
) -> Result<(SocketAddrV4, Mapping), NatPmpError> {
    let gateway = match gateway_override {
        Some(g) => g,
        None => SocketAddr::from((IpAddr::V4(default_gateway()?), NAT_PMP_PORT)),
    };
    let external_ip = request_external_address(gateway, timeout).await?;
    let mapping = request_mapping(gateway, internal_port, lifetime_secs, timeout).await?;
    Ok((
        SocketAddrV4::new(external_ip, mapping.external_port),
        mapping,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requests_follow_rfc_6886() {
        let req = map_request(7000, 7000, 3600);
        assert_eq!(req[0], 0, "version");
        assert_eq!(req[1], 2, "opcode 2 = map TCP");
        assert_eq!(&req[2..4], &[0, 0], "reserved");
        assert_eq!(u16::from_be_bytes([req[4], req[5]]), 7000);
        assert_eq!(u16::from_be_bytes([req[6], req[7]]), 7000);
        assert_eq!(u32::from_be_bytes([req[8], req[9], req[10], req[11]]), 3600);
    }

    #[test]
    fn responses_are_parsed_and_errors_reported() {
        let mut buf = vec![0u8; 16];
        buf[0] = 0;
        buf[1] = 2 + 128;
        buf[8..10].copy_from_slice(&7000u16.to_be_bytes());
        buf[10..12].copy_from_slice(&41234u16.to_be_bytes());
        buf[12..16].copy_from_slice(&3600u32.to_be_bytes());
        let m = parse_map_response(&buf, 7000).unwrap();
        assert_eq!(
            m,
            Mapping {
                external_port: 41234,
                internal_port: 7000,
                lifetime_secs: 3600
            }
        );
        // A mapping for another internal port is refused.
        assert!(matches!(
            parse_map_response(&buf, 7001),
            Err(NatPmpError::Malformed(_))
        ));
        // Result code 2 = network failure.
        let mut refused = buf.clone();
        refused[2..4].copy_from_slice(&2u16.to_be_bytes());
        assert!(matches!(
            parse_map_response(&refused, 7000),
            Err(NatPmpError::Refused(2))
        ));
        // Truncated and wrong-opcode answers are refused.
        assert!(parse_map_response(&buf[..10], 7000).is_err());
        let mut wrong = buf.clone();
        wrong[1] = 1 + 128;
        assert!(parse_map_response(&wrong, 7000).is_err());

        let mut addr = vec![0u8; 12];
        addr[1] = 128;
        addr[8..12].copy_from_slice(&[203, 0, 113, 7]);
        assert_eq!(
            parse_external_address(&addr).unwrap(),
            Ipv4Addr::new(203, 0, 113, 7)
        );
    }

    #[test]
    fn default_gateway_is_read_from_the_routing_table() {
        // Columns: Iface Destination Gateway Flags RefCnt Use Metric Mask MTU Window IRTT
        let table = "Iface\tDestination\tGateway\tFlags\tRefCnt\tUse\tMetric\tMask\tMTU\tWindow\tIRTT\n\
             eth0\t00FFA8C0\t00000000\t0001\t0\t0\t0\t00FFFFFF\t0\t0\t0\n\
             eth0\t00000000\t0101A8C0\t0003\t0\t0\t100\t00000000\t0\t0\t0\n";
        assert_eq!(
            parse_default_gateway(table),
            Some(Ipv4Addr::new(192, 168, 1, 1))
        );
        // A table without a default route yields nothing rather than a wrong answer.
        let no_default = "Iface\tDestination\tGateway\n eth0\t00FFA8C0\t00000000\t0001\n";
        assert_eq!(parse_default_gateway(no_default), None);
    }
}
