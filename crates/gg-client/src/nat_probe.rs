//! `gg-worker nat-probe`: what the local network offers for reachability (ADR-0013).
//!
//! An operator running a contributor behind a home router needs to know, before enabling sharing,
//! whether the gateway hands out port mappings and under which protocol. This probe answers with
//! measurements, never with assumptions: it asks NAT-PMP, then UPnP-IGD, and prints what each
//! answered. With `--map` it takes a **leased** mapping and removes it straight away.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::time::Duration;

use crate::{natpmp, upnp};

/// Run the probe. Returns an error only when the local machine cannot be questioned at all.
pub async fn run(gateway: Option<Ipv4Addr>, map: Option<u16>, lease: u32) -> anyhow::Result<()> {
    let gateway = match gateway {
        Some(g) => g,
        None => match natpmp::default_gateway() {
            Ok(g) => g,
            Err(e) => {
                println!("default gateway: unavailable ({e})");
                println!(
                    "\nno gateway to question: set --gateway, or announce a reachable address"
                );
                return Ok(());
            }
        },
    };
    println!("default gateway: {gateway}");

    // 1. NAT-PMP / PCP port (RFC 6886), read-only request first.
    let pmp_addr = SocketAddr::from((IpAddr::V4(gateway), natpmp::NAT_PMP_PORT));
    match natpmp::request_external_address(pmp_addr, Duration::from_secs(2)).await {
        Ok(ip) => println!("NAT-PMP: answers, external address {ip}"),
        Err(e) => println!("NAT-PMP: {e}"),
    }

    // 2. IPv6: no NAT at all, so nothing to map.
    match global_ipv6() {
        Some(v6) => println!(
            "IPv6: global address {v6} (inbound not verified from here: it needs a host outside this network)"
        ),
        None => println!("IPv6: no global address on this machine"),
    }

    // 3. UPnP-IGD.
    let gw = match upnp::discover(gateway, Duration::from_secs(4)).await {
        Ok(g) => {
            println!("UPnP-IGD: {} at {}", g.service_type, g.control_url);
            match g.external_address().await {
                Ok(ip) => println!("UPnP-IGD: external address {ip}"),
                Err(e) => println!("UPnP-IGD: external address unavailable ({e})"),
            }
            Some(g)
        }
        Err(e) => {
            println!("UPnP-IGD: {e}");
            None
        }
    };

    let Some(port) = map else {
        println!(
            "\nno mapping requested (pass --map <port> to test one; it is deleted immediately)"
        );
        return Ok(());
    };
    let local_ip = local_address_towards(gateway)?;
    let internal = SocketAddr::from((IpAddr::V4(local_ip), port));
    println!("\n-- test mapping of {internal} for {lease}s");
    let mut mapped = false;
    match natpmp::request_mapping(pmp_addr, port, lease, Duration::from_secs(2)).await {
        Ok(m) => {
            println!("NAT-PMP: mapped external port {}", m.external_port);
            mapped = true;
            match natpmp::delete_mapping(pmp_addr, port, Duration::from_secs(2)).await {
                Ok(()) => println!("NAT-PMP: mapping deleted"),
                Err(e) => {
                    println!("NAT-PMP: could not delete the mapping ({e}) — it expires in {lease}s")
                }
            }
        }
        Err(e) => println!("NAT-PMP: no mapping ({e})"),
    }
    if let Some(gw) = gw {
        match gw
            .add_port_mapping(internal, port, lease, "gpu-goblin nat-probe")
            .await
        {
            Ok(()) => {
                println!("UPnP-IGD: mapped external port {port} to {internal}");
                mapped = true;
                // Read it back from the gateway rather than trusting the request.
                match gw.get_port_mapping(port).await {
                    Ok(Some((client, lease_left))) => println!(
                        "UPnP-IGD: gateway confirms {port} -> {client}, lease {lease_left}s"
                    ),
                    Ok(None) => {
                        println!("UPnP-IGD: gateway reports no such mapping (it did not take)")
                    }
                    Err(e) => println!("UPnP-IGD: could not read the mapping back ({e})"),
                }
                match gw.delete_port_mapping(port).await {
                    Ok(()) => match gw.get_port_mapping(port).await {
                        Ok(None) => {
                            println!("UPnP-IGD: mapping deleted, gateway confirms it is gone")
                        }
                        Ok(Some(_)) => {
                            println!("UPnP-IGD: deletion accepted but the entry is still there")
                        }
                        Err(e) => println!("UPnP-IGD: mapping deleted, read-back failed ({e})"),
                    },
                    Err(e) => println!(
                        "UPnP-IGD: could not delete the mapping ({e}) — it expires in {lease}s"
                    ),
                }
            }
            Err(e) => println!("UPnP-IGD: no mapping ({e})"),
        }
    }
    if !mapped {
        println!(
            "\nno protocol produced a mapping: announce a forwarded address with p2p.advertise_addr, \
             or let the worker fetch from peers without sharing"
        );
    }
    Ok(())
}

/// Global IPv6 address this machine would use to reach the Internet, if it has one. Link-local,
/// unique-local and loopback addresses are not global and are never returned: over IPv6 there is no
/// NAT, so a global address is directly reachable once the firewall allows it.
pub fn global_ipv6() -> Option<std::net::Ipv6Addr> {
    // Connecting a UDP socket sends nothing; it only makes the kernel pick a route and a source.
    let socket = std::net::UdpSocket::bind("[::]:0").ok()?;
    socket
        .connect((
            std::net::Ipv6Addr::new(0x2001, 0x4860, 0x4860, 0, 0, 0, 0, 0x8888),
            9,
        ))
        .ok()?;
    match socket.local_addr().ok()?.ip() {
        IpAddr::V6(v6) => is_global_ipv6(v6).then_some(v6),
        IpAddr::V4(_) => None,
    }
}

/// Whether an IPv6 address is globally routable for our purposes.
pub fn is_global_ipv6(v6: std::net::Ipv6Addr) -> bool {
    if v6.is_loopback() || v6.is_unspecified() || v6.is_multicast() {
        return false;
    }
    let first = v6.segments()[0];
    let link_local = first & 0xffc0 == 0xfe80;
    let unique_local = first & 0xfe00 == 0xfc00;
    let documentation = v6.segments()[0] == 0x2001 && v6.segments()[1] == 0x0db8;
    // IPv4-mapped (::ffff:a.b.c.d) is an IPv4 address in disguise, not a v6 one.
    let v4_mapped = v6.to_ipv4_mapped().is_some();
    !(link_local || unique_local || documentation || v4_mapped)
}

/// Local address used to reach the gateway (no packet is sent: a connected UDP socket just picks
/// the route).
pub fn local_address_towards(gateway: Ipv4Addr) -> anyhow::Result<Ipv4Addr> {
    let socket = std::net::UdpSocket::bind((Ipv4Addr::UNSPECIFIED, 0))?;
    socket.connect((gateway, 9))?;
    match socket.local_addr()?.ip() {
        IpAddr::V4(v4) => Ok(v4),
        IpAddr::V6(_) => anyhow::bail!("no IPv4 address towards the gateway"),
    }
}
