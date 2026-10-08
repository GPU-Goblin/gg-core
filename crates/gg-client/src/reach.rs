//! How a chunk server becomes reachable (ADR-0013; ADR-0041 A8), shared by the worker and the lab
//! node. The candidate addresses come in order of preference, and the control plane is the judge:
//! it dials each one back before advertising it, so an address nobody can connect to is never
//! given to anyone.
//!
//! 1. an address configured by the person who runs the machine;
//! 2. a global IPv6 address (no NAT to traverse);
//! 3. a port mapped on the gateway, by NAT-PMP, then UPnP-IGD (removed again on [`Reach::release`]);
//! 4. the address the chunk server is bound to, when concrete (a laboratory network, one host);
//! 5. the source address the control plane observes (a NAT that preserves the port).

use std::net::{IpAddr, SocketAddr};
use std::time::{Duration, Instant};

use tokio::sync::Mutex;

/// What the machine allows.
#[derive(Clone, Debug)]
pub struct ReachConfig {
    /// Address to announce, configured; port 0 means "the bound port". Empty: discover.
    pub advertise_addr: String,
    /// Offer a global IPv6 address when the chunk server listens on IPv6 or on every address.
    pub ipv6: bool,
    /// Ask the gateway for a mapping with NAT-PMP.
    pub nat_pmp: bool,
    /// Ask the gateway for a mapping with UPnP-IGD.
    pub upnp: bool,
    /// Gateway to ask (`ip:port`); empty: the IPv4 default route.
    pub gateway: String,
    /// Lease requested for a mapping, in seconds.
    pub lease_secs: u32,
}

impl Default for ReachConfig {
    fn default() -> Self {
        Self {
            advertise_addr: String::new(),
            ipv6: true,
            nat_pmp: true,
            upnp: true,
            gateway: String::new(),
            lease_secs: 3600,
        }
    }
}

/// An address the machine could be reached at, with how it was obtained and when it must be
/// resolved again.
#[derive(Clone, Debug)]
pub struct Candidate {
    /// `ip:port` (IPv6 in brackets).
    pub addr: String,
    /// `configured`, `ipv6`, `nat-pmp`, `upnp`, `bound` or `observed`.
    pub how: &'static str,
    /// When this address must be resolved again.
    pub renew: Instant,
}

impl Candidate {
    fn new(addr: String, how: &'static str, valid_for: Duration) -> Self {
        Self {
            addr,
            how,
            renew: Instant::now() + valid_for,
        }
    }
}

/// The reachability of one chunk server: its candidates, the address the control plane accepted,
/// and the gateway mapping to remove at the end.
pub struct Reach {
    cfg: ReachConfig,
    accepted: Mutex<Option<(String, &'static str, Instant)>>,
    upnp_mapping: Mutex<Option<(crate::upnp::Gateway, u16)>>,
}

impl Reach {
    pub fn new(cfg: ReachConfig) -> Self {
        Self {
            cfg,
            accepted: Mutex::new(None),
            upnp_mapping: Mutex::new(None),
        }
    }

    /// The address accepted last and how it was obtained, while it is still valid.
    pub async fn current(&self) -> Option<(String, &'static str)> {
        match self.accepted.lock().await.as_ref() {
            Some((addr, how, renew)) if Instant::now() < *renew => Some((addr.clone(), how)),
            _ => None,
        }
    }

    /// Forget the address accepted: the next announcement resolves one again.
    pub async fn forget(&self) {
        *self.accepted.lock().await = None;
    }

    /// Announce through `try_addr` (which asks the control plane to record an address and returns
    /// whether it accepted it): the address accepted before while still valid, otherwise each
    /// candidate in turn until one is accepted. `observed`: the source address the control plane
    /// sees, asked for only if needed. Returns the address accepted.
    pub async fn announce<T, TF, O>(
        &self,
        bound: SocketAddr,
        observed: O,
        try_addr: T,
    ) -> Option<(String, &'static str)>
    where
        T: Fn(String) -> TF,
        TF: std::future::Future<Output = Result<(), String>>,
        O: std::future::Future<Output = Option<IpAddr>>,
    {
        if let Some((addr, how)) = self.current().await {
            match try_addr(addr.clone()).await {
                Ok(()) => return Some((addr, how)),
                Err(e) => {
                    tracing::debug!(%addr, error = %e, "announce refused, resolving the address again");
                    self.forget().await;
                }
            }
        }
        for c in self.candidates(bound, observed).await {
            match try_addr(c.addr.clone()).await {
                Ok(()) => {
                    tracing::info!(addr = %c.addr, how = c.how, "announced a reachable address");
                    *self.accepted.lock().await = Some((c.addr.clone(), c.how, c.renew));
                    return Some((c.addr, c.how));
                }
                Err(e) => tracing::info!(
                    addr = %c.addr,
                    how = c.how,
                    error = %e,
                    "the control plane could not use this address, trying the next one"
                ),
            }
        }
        tracing::info!("no reachable address for peer sharing");
        None
    }

    /// The candidates, in order of preference.
    pub async fn candidates<O>(&self, bound: SocketAddr, observed: O) -> Vec<Candidate>
    where
        O: std::future::Future<Output = Option<IpAddr>>,
    {
        let far = Duration::from_secs(3600);
        let mut out = Vec::new();

        // 1. What the person who runs the machine configured wins over anything discovered.
        if !self.cfg.advertise_addr.is_empty()
            && let Ok(a) = self.cfg.advertise_addr.parse::<SocketAddr>()
        {
            let addr = if a.port() == 0 {
                SocketAddr::new(a.ip(), bound.port()).to_string()
            } else {
                a.to_string()
            };
            out.push(Candidate::new(addr, "configured", far));
        }

        // 2. IPv6 first when the chunk server can answer on it: there is no NAT to traverse, so no
        // mapping to obtain, renew or clean up. The control plane still dials the address back,
        // which is what catches a firewall that blocks incoming IPv6.
        if self.cfg.ipv6
            && (bound.is_ipv6() || bound.ip().is_unspecified())
            && let Some(v6) = crate::nat_probe::global_ipv6()
        {
            out.push(Candidate::new(
                SocketAddr::new(IpAddr::V6(v6), bound.port()).to_string(),
                "ipv6",
                far,
            ));
        }

        // 3. The port-mapping protocols, for IPv4 behind a NAT. A chunk server listening on
        // loopback only cannot be reached through the gateway: a mapping would lead nowhere.
        let routable = !bound.ip().is_loopback();
        if self.cfg.nat_pmp && routable {
            let gateway = self.cfg.gateway.parse::<SocketAddr>().ok();
            match crate::natpmp::map_port(
                bound.port(),
                self.cfg.lease_secs,
                Duration::from_secs(2),
                gateway,
            )
            .await
            {
                Ok((external, mapping)) => {
                    tracing::info!(
                        %external,
                        lifetime_secs = mapping.lifetime_secs,
                        "gateway mapped a port for peer sharing (NAT-PMP)"
                    );
                    // Renew halfway through the granted lifetime, as RFC 6886 advises.
                    out.push(Candidate::new(
                        external.to_string(),
                        "nat-pmp",
                        Duration::from_secs((mapping.lifetime_secs.max(120) / 2) as u64),
                    ));
                }
                Err(e) => tracing::debug!(error = %e, "no NAT-PMP mapping"),
            }
        }
        if self.cfg.upnp
            && routable
            && let Some(c) = self.map_port_with_upnp(bound).await
        {
            out.push(c);
        }

        // 4. A concrete bound address (laboratory network, single host).
        if !bound.ip().is_unspecified() {
            out.push(Candidate::new(bound.to_string(), "bound", far));
        }

        // 5. Last, the source address the control plane sees. Behind a NAT that preserves the
        // port this is the reachable one; over IPv6 it is simply this machine.
        if let Some(ip) = observed.await {
            let addr = SocketAddr::new(ip, bound.port()).to_string();
            if !out.iter().any(|c| c.addr == addr) {
                out.push(Candidate::new(addr, "observed", Duration::from_secs(600)));
            }
        }
        out
    }

    /// Ask the gateway for a UPnP-IGD mapping and check it took. Any failure returns `None`.
    async fn map_port_with_upnp(&self, bound: SocketAddr) -> Option<Candidate> {
        let gateway_ip = match self.cfg.gateway.parse::<SocketAddr>() {
            Ok(a) => match a.ip() {
                IpAddr::V4(v4) => v4,
                IpAddr::V6(_) => return None,
            },
            Err(_) => crate::natpmp::default_gateway().ok()?,
        };
        let gw = match crate::upnp::discover(gateway_ip, Duration::from_secs(4)).await {
            Ok(g) => g,
            Err(e) => {
                tracing::info!(error = %e, "no UPnP gateway");
                return None;
            }
        };
        let local = crate::nat_probe::local_address_towards(gateway_ip).ok()?;
        let internal = SocketAddr::from((IpAddr::V4(local), bound.port()));
        let lease = self.cfg.lease_secs;
        if let Err(e) = gw
            .add_port_mapping(internal, bound.port(), lease, "gpu-goblin peer")
            .await
        {
            tracing::info!(error = %e, "gateway refused a UPnP mapping");
            return None;
        }
        // Trust the gateway's own table, not our request.
        match gw.get_port_mapping(bound.port()).await {
            Ok(Some(_)) => {}
            Ok(None) => {
                tracing::info!("the gateway accepted a UPnP mapping but does not list it");
                return None;
            }
            Err(e) => tracing::debug!(error = %e, "could not read the UPnP mapping back"),
        }
        let external = gw.external_address().await.ok()?;
        *self.upnp_mapping.lock().await = Some((gw, bound.port()));
        let addr = format!("{external}:{}", bound.port());
        tracing::info!(%addr, lease_secs = lease, "gateway mapped a port for peer sharing (UPnP-IGD)");
        Some(Candidate::new(
            addr,
            "upnp",
            Duration::from_secs((lease.max(120) / 2) as u64),
        ))
    }

    /// Remove the gateway mapping this machine created: stopping leaves no hole behind.
    pub async fn release(&self) {
        if let Some((gw, port)) = self.upnp_mapping.lock().await.take() {
            match gw.delete_port_mapping(port).await {
                Ok(()) => tracing::info!(port, "UPnP mapping deleted"),
                Err(e) => {
                    tracing::warn!(port, error = %e, "could not delete the UPnP mapping; it expires with its lease")
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    fn quiet(advertise: &str) -> Reach {
        Reach::new(ReachConfig {
            advertise_addr: advertise.into(),
            ipv6: false,
            nat_pmp: false,
            upnp: false,
            gateway: String::new(),
            lease_secs: 120,
        })
    }

    #[tokio::test]
    async fn candidates_come_in_order_and_an_observed_duplicate_is_dropped() {
        let bound: SocketAddr = "192.168.1.20:7451".parse().unwrap();
        let r = quiet("203.0.113.9:0");
        let c = r
            .candidates(bound, async { Some("198.51.100.7".parse().unwrap()) })
            .await;
        let got: Vec<(&str, &str)> = c.iter().map(|c| (c.addr.as_str(), c.how)).collect();
        assert_eq!(
            got,
            vec![
                ("203.0.113.9:7451", "configured"),
                ("192.168.1.20:7451", "bound"),
                ("198.51.100.7:7451", "observed"),
            ]
        );
        // The observed address equal to the bound one is not offered twice.
        let c = quiet("")
            .candidates(bound, async { Some("192.168.1.20".parse().unwrap()) })
            .await;
        assert_eq!(c.len(), 1);
        // Loopback only: no gateway is asked (it would map a port leading nowhere).
        let r = Reach::new(ReachConfig {
            ipv6: false,
            gateway: "127.0.0.1:1".into(),
            ..Default::default()
        });
        let c = r
            .candidates("127.0.0.1:7451".parse().unwrap(), async { None })
            .await;
        assert_eq!(c.iter().map(|c| c.how).collect::<Vec<_>>(), vec!["bound"]);
    }

    #[tokio::test]
    async fn the_accepted_address_is_kept_until_the_control_plane_refuses_it() {
        let bound: SocketAddr = "10.0.0.5:7451".parse().unwrap();
        let r = quiet("");
        let tried = StdMutex::new(Vec::<String>::new());
        let refuse = StdMutex::new(vec!["10.0.0.5:7451".to_owned()]);
        let try_addr = |a: String| {
            tried.lock().unwrap().push(a.clone());
            let refused = refuse.lock().unwrap().contains(&a);
            async move {
                if refused {
                    Err("not reachable".into())
                } else {
                    Ok(())
                }
            }
        };
        // The bound address is refused (the dial-back failed), the observed one accepted.
        let observed = || async { Some("198.51.100.7".parse().unwrap()) };
        let got = r.announce(bound, observed(), try_addr).await;
        assert_eq!(got, Some(("198.51.100.7:7451".into(), "observed")));
        // Next time it is tried directly, nothing else.
        tried.lock().unwrap().clear();
        r.announce(bound, observed(), try_addr).await;
        assert_eq!(*tried.lock().unwrap(), vec!["198.51.100.7:7451".to_owned()]);
        // Refused later (the NAT changed): resolved again, and nothing left to accept.
        refuse.lock().unwrap().push("198.51.100.7:7451".into());
        assert_eq!(r.announce(bound, observed(), try_addr).await, None);
        assert_eq!(r.current().await, None);
    }
}
