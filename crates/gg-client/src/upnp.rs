//! UPnP Internet Gateway Device client (ADR-0013) used to obtain a port mapping from a home router.
//!
//! Measured need: the gateway of the development network (a consumer ISP router)
//! answers neither NAT-PMP nor PCP on UDP 5351, but does expose an `InternetGatewayDevice` with a
//! `WANIPConnection` service. This client discovers that service, reads the external address and
//! asks for a **leased** port mapping, which it deletes when sharing stops.
//!
//! Two rules keep the discovery from becoming a way to make this worker talk to anything:
//! * only the **default gateway** (or an explicitly configured address) is ever contacted: an SSDP
//!   answer from any other device on the network is ignored;
//! * the device description is read with a bounded, tag-scanning reader that extracts a handful of
//!   known elements. Nothing in the document is executed, resolved or followed elsewhere.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::time::Duration;

/// Multicast address of SSDP discovery.
const SSDP_ADDR: (Ipv4Addr, u16) = (Ipv4Addr::new(239, 255, 255, 250), 1900);
/// Largest device description accepted.
const MAX_DESCRIPTION_BYTES: usize = 256 * 1024;
/// Services able to create a port mapping, most specific first.
const WAN_SERVICES: [&str; 2] = [
    "urn:schemas-upnp-org:service:WANIPConnection:1",
    "urn:schemas-upnp-org:service:WANPPPConnection:1",
];

/// Failures that all leave peer sharing working through another address.
#[derive(Debug, thiserror::Error)]
pub enum UpnpError {
    /// No gateway answered the discovery.
    #[error("no UPnP gateway answered")]
    NotFound,
    /// The description had no usable WAN connection service.
    #[error("gateway exposes no WANIPConnection/WANPPPConnection service")]
    NoService,
    /// The gateway refused the request (SOAP fault).
    #[error("gateway refused the request: {code} {description}")]
    Refused {
        /// UPnP error code.
        code: u16,
        /// Error description.
        description: String,
    },
    /// Malformed answer.
    #[error("malformed answer: {0}")]
    Malformed(String),
    /// Transport.
    #[error("http: {0}")]
    Http(String),
    /// Local I/O.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

/// A discovered gateway service able to map ports.
#[derive(Clone, Debug)]
pub struct Gateway {
    /// Absolute SOAP control URL.
    pub control_url: String,
    /// Service type used in SOAP actions.
    pub service_type: String,
    /// Address the gateway answered from.
    pub address: IpAddr,
}

/// Extract the text of the first `<tag>` in `xml`, bounded and without interpreting anything else.
pub fn tag_text(xml: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = xml.find(&open)? + open.len();
    let end = xml[start..].find(&close)? + start;
    let text = xml[start..end].trim();
    if text.is_empty() || text.len() > 512 {
        return None;
    }
    Some(text.to_owned())
}

/// Find the control URL of the first WAN connection service in a device description.
pub fn find_wan_service(xml: &str) -> Option<(String, String)> {
    for wanted in WAN_SERVICES {
        let mut rest = xml;
        while let Some(start) = rest.find("<service>") {
            let block_start = start + "<service>".len();
            let Some(end) = rest[block_start..].find("</service>") else {
                break;
            };
            let block = &rest[block_start..block_start + end];
            if let (Some(t), Some(c)) = (
                tag_text(block, "serviceType"),
                tag_text(block, "controlURL"),
            ) && t == wanted
            {
                return Some((t, c));
            }
            rest = &rest[block_start + end..];
        }
    }
    None
}

/// Absolute URL of a control path relative to the description URL.
pub fn absolute_url(description_url: &str, control_path: &str) -> Option<String> {
    if control_path.starts_with("http://") || control_path.starts_with("https://") {
        return Some(control_path.to_owned());
    }
    let after_scheme = description_url.strip_prefix("http://")?;
    let authority = after_scheme.split('/').next()?;
    if control_path.starts_with('/') {
        Some(format!("http://{authority}{control_path}"))
    } else {
        Some(format!("http://{authority}/{control_path}"))
    }
}

/// Read the UPnP error code of a SOAP fault, if the answer is one.
pub fn soap_fault(xml: &str) -> Option<(u16, String)> {
    if !xml.contains("<s:Fault>") && !xml.contains("<SOAP-ENV:Fault>") {
        return None;
    }
    let code = tag_text(xml, "errorCode")
        .and_then(|c| c.parse().ok())
        .unwrap_or(0);
    let description = tag_text(xml, "errorDescription").unwrap_or_else(|| "unknown".into());
    Some((code, description))
}

/// Discover the gateway through SSDP, contacting only `gateway_ip`.
pub async fn discover(gateway_ip: Ipv4Addr, timeout: Duration) -> Result<Gateway, UpnpError> {
    let socket = tokio::net::UdpSocket::bind(SocketAddr::from((Ipv4Addr::UNSPECIFIED, 0))).await?;
    let search = format!(
        "M-SEARCH * HTTP/1.1\r\nHOST: {}:{}\r\nMAN: \"ssdp:discover\"\r\nMX: 2\r\nST: urn:schemas-upnp-org:device:InternetGatewayDevice:1\r\n\r\n",
        SSDP_ADDR.0, SSDP_ADDR.1
    );
    socket
        .send_to(search.as_bytes(), SocketAddr::from(SSDP_ADDR))
        .await?;
    let deadline = tokio::time::Instant::now() + timeout;
    let mut buf = vec![0u8; 4096];
    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            return Err(UpnpError::NotFound);
        }
        let Ok(Ok((n, from))) = tokio::time::timeout(remaining, socket.recv_from(&mut buf)).await
        else {
            return Err(UpnpError::NotFound);
        };
        // Only the gateway is trusted: another device on the network cannot point us elsewhere.
        if from.ip() != IpAddr::V4(gateway_ip) {
            tracing::debug!(%from, "ignoring an SSDP answer from a device that is not the gateway");
            continue;
        }
        let text = String::from_utf8_lossy(&buf[..n]);
        let Some(location) = text
            .lines()
            .find_map(|l| {
                l.strip_prefix("LOCATION:")
                    .or_else(|| l.strip_prefix("Location:"))
            })
            .map(|l| l.trim().to_owned())
        else {
            continue;
        };
        if !location.starts_with(&format!("http://{gateway_ip}:"))
            && !location.starts_with(&format!("http://{gateway_ip}/"))
        {
            tracing::debug!(%location, "ignoring a description URL outside the gateway");
            continue;
        }
        return describe(&location, IpAddr::V4(gateway_ip)).await;
    }
}

/// Fetch and read a device description.
async fn describe(location: &str, address: IpAddr) -> Result<Gateway, UpnpError> {
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .map_err(|e| UpnpError::Http(e.to_string()))?;
    let r = http
        .get(location)
        .send()
        .await
        .map_err(|e| UpnpError::Http(e.to_string()))?;
    let bytes = r
        .bytes()
        .await
        .map_err(|e| UpnpError::Http(e.to_string()))?;
    if bytes.len() > MAX_DESCRIPTION_BYTES {
        return Err(UpnpError::Malformed("device description too large".into()));
    }
    let xml = String::from_utf8_lossy(&bytes);
    let (service_type, control_path) = find_wan_service(&xml).ok_or(UpnpError::NoService)?;
    let control_url = absolute_url(location, &control_path)
        .ok_or_else(|| UpnpError::Malformed("control URL is not usable".into()))?;
    Ok(Gateway {
        control_url,
        service_type,
        address,
    })
}

impl Gateway {
    /// Send one SOAP action and return the body.
    async fn soap(&self, action: &str, arguments: &str) -> Result<String, UpnpError> {
        let body = format!(
            "<?xml version=\"1.0\"?><s:Envelope xmlns:s=\"http://schemas.xmlsoap.org/soap/envelope/\" s:encodingStyle=\"http://schemas.xmlsoap.org/soap/encoding/\"><s:Body><u:{action} xmlns:u=\"{}\">{arguments}</u:{action}></s:Body></s:Envelope>",
            self.service_type
        );
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .map_err(|e| UpnpError::Http(e.to_string()))?;
        let r = http
            .post(&self.control_url)
            .header("content-type", "text/xml; charset=\"utf-8\"")
            .header("soapaction", format!("\"{}#{action}\"", self.service_type))
            .body(body)
            .send()
            .await
            .map_err(|e| UpnpError::Http(e.to_string()))?;
        let status = r.status();
        let text = r.text().await.map_err(|e| UpnpError::Http(e.to_string()))?;
        if let Some((code, description)) = soap_fault(&text) {
            return Err(UpnpError::Refused { code, description });
        }
        if !status.is_success() {
            return Err(UpnpError::Http(format!("gateway answered {status}")));
        }
        Ok(text)
    }

    /// Address the gateway presents to the outside.
    pub async fn external_address(&self) -> Result<Ipv4Addr, UpnpError> {
        let text = self.soap("GetExternalIPAddress", "").await?;
        tag_text(&text, "NewExternalIPAddress")
            .and_then(|a| a.parse().ok())
            .ok_or_else(|| UpnpError::Malformed("no external address in the answer".into()))
    }

    /// Ask for a leased TCP mapping from `external_port` to `internal`. The lease matters: a worker
    /// that disappears does not leave a hole open for ever.
    pub async fn add_port_mapping(
        &self,
        internal: SocketAddr,
        external_port: u16,
        lease_secs: u32,
        description: &str,
    ) -> Result<(), UpnpError> {
        let args = format!(
            "<NewRemoteHost></NewRemoteHost><NewExternalPort>{external_port}</NewExternalPort><NewProtocol>TCP</NewProtocol><NewInternalPort>{}</NewInternalPort><NewInternalClient>{}</NewInternalClient><NewEnabled>1</NewEnabled><NewPortMappingDescription>{}</NewPortMappingDescription><NewLeaseDuration>{lease_secs}</NewLeaseDuration>",
            internal.port(),
            internal.ip(),
            sanitize(description)
        );
        self.soap("AddPortMapping", &args).await?;
        Ok(())
    }

    /// Read back a mapping by external port. `Ok(None)` means the gateway has no such entry: this
    /// is how a worker checks that a mapping was really created, and really removed, instead of
    /// trusting its own request.
    pub async fn get_port_mapping(
        &self,
        external_port: u16,
    ) -> Result<Option<(SocketAddr, u32)>, UpnpError> {
        let args = format!(
            "<NewRemoteHost></NewRemoteHost><NewExternalPort>{external_port}</NewExternalPort><NewProtocol>TCP</NewProtocol>"
        );
        match self.soap("GetSpecificPortMappingEntry", &args).await {
            Ok(text) => {
                let client = tag_text(&text, "NewInternalClient")
                    .and_then(|c| c.parse::<Ipv4Addr>().ok())
                    .ok_or_else(|| {
                        UpnpError::Malformed("no internal client in the answer".into())
                    })?;
                let port = tag_text(&text, "NewInternalPort")
                    .and_then(|p| p.parse::<u16>().ok())
                    .ok_or_else(|| UpnpError::Malformed("no internal port in the answer".into()))?;
                let lease = tag_text(&text, "NewLeaseDuration")
                    .and_then(|l| l.parse::<u32>().ok())
                    .unwrap_or(0);
                Ok(Some((SocketAddr::from((IpAddr::V4(client), port)), lease)))
            }
            // 714 NoSuchEntryInArray: the entry does not exist, which is an answer, not a failure.
            Err(UpnpError::Refused { code: 714, .. }) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Remove a mapping this worker created.
    pub async fn delete_port_mapping(&self, external_port: u16) -> Result<(), UpnpError> {
        let args = format!(
            "<NewRemoteHost></NewRemoteHost><NewExternalPort>{external_port}</NewExternalPort><NewProtocol>TCP</NewProtocol>"
        );
        self.soap("DeletePortMapping", &args).await?;
        Ok(())
    }
}

/// Keep a description to plain characters: it ends up in a router's table and in its interface.
fn sanitize(text: &str) -> String {
    text.chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, ' ' | '-' | '_' | '.'))
        .take(64)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const DESCRIPTION: &str = r#"<?xml version="1.0"?>
<root xmlns="urn:schemas-upnp-org:device-1-0"><device>
<deviceType>urn:schemas-upnp-org:device:InternetGatewayDevice:1</deviceType>
<serviceList><service>
  <serviceType>urn:schemas-upnp-org:service:Layer3Forwarding:1</serviceType>
  <controlURL>/upnp/control/L3F</controlURL>
</service></serviceList>
<deviceList><device><serviceList>
<service>
  <serviceType>urn:schemas-upnp-org:service:WANCommonInterfaceConfig:1</serviceType>
  <controlURL>/igd/upnp/control/WANCommonIFC1</controlURL>
</service>
<service>
  <serviceType>urn:schemas-upnp-org:service:WANIPConnection:1</serviceType>
  <controlURL>/igd/upnp/control/WANIPConn1</controlURL>
</service>
</serviceList></device></deviceList>
</device></root>"#;

    #[test]
    fn the_wan_service_is_found_in_a_real_description() {
        let (service, control) = find_wan_service(DESCRIPTION).unwrap();
        assert_eq!(service, "urn:schemas-upnp-org:service:WANIPConnection:1");
        assert_eq!(control, "/igd/upnp/control/WANIPConn1");
        assert_eq!(
            absolute_url("http://192.168.1.1:60000/igd/gatedesc1.xml", &control).unwrap(),
            "http://192.168.1.1:60000/igd/upnp/control/WANIPConn1"
        );
        // A description without a WAN service yields nothing rather than a wrong guess.
        assert!(find_wan_service("<root><device></device></root>").is_none());
        assert!(
            find_wan_service("<service><serviceType>urn:other:1</serviceType></service>").is_none()
        );
        // Truncated documents do not panic.
        for cut in [10, 120, 400] {
            let _ = find_wan_service(&DESCRIPTION[..cut.min(DESCRIPTION.len())]);
        }
    }

    #[test]
    fn answers_and_faults_are_read_without_trusting_the_document() {
        let ok = "<s:Envelope><s:Body><u:GetExternalIPAddressResponse><NewExternalIPAddress>203.0.113.7</NewExternalIPAddress></u:GetExternalIPAddressResponse></s:Body></s:Envelope>";
        assert_eq!(
            tag_text(ok, "NewExternalIPAddress")
                .unwrap()
                .parse::<Ipv4Addr>()
                .unwrap(),
            Ipv4Addr::new(203, 0, 113, 7)
        );
        assert!(soap_fault(ok).is_none());
        let fault = "<s:Envelope><s:Body><s:Fault><detail><UPnPError><errorCode>718</errorCode><errorDescription>ConflictInMappingEntry</errorDescription></UPnPError></detail></s:Fault></s:Body></s:Envelope>";
        let (code, description) = soap_fault(fault).unwrap();
        assert_eq!(code, 718);
        assert_eq!(description, "ConflictInMappingEntry");
        // Oversized or empty values are refused rather than propagated.
        let huge = format!("<a>{}</a>", "x".repeat(600));
        assert!(tag_text(&huge, "a").is_none());
        assert!(tag_text("<a></a>", "a").is_none());
        assert!(tag_text("<a>unclosed", "a").is_none());
    }

    #[test]
    fn mapping_descriptions_stay_plain() {
        assert_eq!(sanitize("gpu-goblin peer 1"), "gpu-goblin peer 1");
        assert_eq!(sanitize("<script>alert(1)</script>"), "scriptalert1script");
        assert_eq!(sanitize(&"a".repeat(200)).len(), 64);
    }
}
