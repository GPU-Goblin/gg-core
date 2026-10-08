//! What the owner of a worker allows, and what the worker reports (ADR-0043): the data of the
//! local control API between `gg-worker` and the contributor's application. The worker enforces it;
//! the application edits and shows it.

use std::path::Path;

use serde::{Deserialize, Serialize};

/// Utilization of the card, by others, above which the worker yields it: a game or a render, not a
/// desktop (an idle desktop measured 17 % on the development machine's card).
pub const OTHER_GPU_USE_PERCENT: u32 = 50;

/// A window of days and hours in which the worker may compute (local time).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Window {
    /// Days, `mon` … `sun`; empty: every day.
    #[serde(default)]
    pub days: Vec<String>,
    /// Start, `HH:MM`.
    pub from: String,
    /// End, `HH:MM`; before `from`, the window runs past midnight.
    pub to: String,
}

/// What the owner allows.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Policy {
    /// Lend nothing until resumed.
    pub paused: bool,
    /// Lend only while the session is locked, or idle for `idle_after_secs`.
    pub only_when_idle: bool,
    /// Idle time before lending, when `only_when_idle`.
    pub idle_after_secs: u64,
    /// Windows in which lending is allowed; empty: at any time.
    pub schedule: Vec<Window>,
    /// Claim nothing while the card is hotter than this (°C).
    pub gpu_max_temp_c: Option<u32>,
    /// Claim nothing while another program uses the card (a game, a render).
    pub yield_to_other_gpu_apps: bool,
    /// Host memory lent to a task, bytes (at most the configuration's).
    pub max_ram_bytes: Option<u64>,
    /// Accelerator memory lent to a task, bytes (at most the configuration's).
    pub device_memory_bytes: Option<u64>,
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            paused: false,
            only_when_idle: false,
            idle_after_secs: 300,
            schedule: Vec::new(),
            gpu_max_temp_c: None,
            yield_to_other_gpu_apps: true,
            max_ram_bytes: None,
            device_memory_bytes: None,
        }
    }
}

const DAYS: [&str; 7] = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"];

fn minutes(hhmm: &str) -> Option<u32> {
    let (h, m) = hhmm.split_once(':')?;
    let (h, m): (u32, u32) = (h.parse().ok()?, m.parse().ok()?);
    (h < 24 && m < 60).then_some(h * 60 + m)
}

impl Policy {
    /// Refuse what cannot be meant.
    pub fn validate(&self) -> Result<(), String> {
        if self.idle_after_secs > 24 * 3600 {
            return Err("idle_after_secs above a day".into());
        }
        if self.schedule.len() > 64 {
            return Err("more than 64 windows".into());
        }
        for w in &self.schedule {
            if minutes(&w.from).is_none() || minutes(&w.to).is_none() {
                return Err(format!("window {}–{}: times are HH:MM", w.from, w.to));
            }
            if let Some(d) = w.days.iter().find(|d| !DAYS.contains(&d.as_str())) {
                return Err(format!("unknown day '{d}' (mon … sun)"));
            }
        }
        if self
            .gpu_max_temp_c
            .is_some_and(|t| !(40..=110).contains(&t))
        {
            return Err("gpu_max_temp_c between 40 and 110".into());
        }
        if self.max_ram_bytes == Some(0) || self.device_memory_bytes == Some(0) {
            return Err("a memory limit must be positive".into());
        }
        Ok(())
    }

    /// Read `path`; the defaults when it does not exist.
    pub fn load(path: &Path) -> Result<Self, String> {
        match std::fs::read_to_string(path) {
            Ok(text) => {
                let p: Policy =
                    toml::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
                p.validate()?;
                Ok(p)
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(format!("{}: {e}", path.display())),
        }
    }

    /// Write `path` atomically.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        self.validate()?;
        let io = |e: std::io::Error| format!("{}: {e}", path.display());
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(io)?;
        }
        let tmp = path.with_extension("toml.tmp");
        let text = toml::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(&tmp, text).map_err(io)?;
        std::fs::rename(&tmp, path).map_err(io)?;
        Ok(())
    }

    /// Whether `now` (day of the week, 0 = Monday, and minutes since midnight) is in a window.
    pub fn in_schedule(&self, day: u32, minute: u32) -> bool {
        if self.schedule.is_empty() {
            return true;
        }
        let name = |d: u32| DAYS[(d % 7) as usize];
        self.schedule.iter().any(|w| {
            let (Some(from), Some(to)) = (minutes(&w.from), minutes(&w.to)) else {
                return false;
            };
            let on = |d: u32| w.days.is_empty() || w.days.iter().any(|x| x == name(d));
            if from <= to {
                on(day) && (from..to).contains(&minute)
            } else {
                // Past midnight: the evening of `day`, or the morning after the day before.
                (on(day) && minute >= from) || (on((day + 6) % 7) && minute < to)
            }
        })
    }
}

/// What the machine looks like right now. `None`: not known (no session, no such card).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Readings {
    /// The session is locked.
    pub locked: Option<bool>,
    /// Seconds since the session became idle; 0 when in use.
    pub idle_secs: Option<u64>,
    /// Card temperature, °C.
    pub gpu_temp_c: Option<u32>,
    /// Card utilization, percent (the worker's own work included).
    pub gpu_busy_percent: Option<u32>,
}

/// Why the worker does not lend right now.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "reason", rename_all = "snake_case")]
pub enum Waiting {
    /// The owner paused it.
    Paused,
    /// Outside the hours allowed.
    OutsideSchedule,
    /// The owner is using the machine.
    NotIdle,
    /// The card is too hot.
    GpuHot { temp_c: u32, max_c: u32 },
    /// Another program uses the card.
    GpuInUse { percent: u32 },
}

/// Whether a new task may be claimed, `readings` taken while the worker computes nothing.
pub fn gate(p: &Policy, r: &Readings, day: u32, minute: u32) -> Option<Waiting> {
    if p.paused {
        return Some(Waiting::Paused);
    }
    if !p.in_schedule(day, minute) {
        return Some(Waiting::OutsideSchedule);
    }
    if p.only_when_idle {
        // A machine without a session (a server, a box without a screen) is idle.
        let idle = r.locked == Some(true)
            || r.idle_secs.is_none_or(|s| s >= p.idle_after_secs) && r.locked.is_none()
            || r.idle_secs.is_some_and(|s| s >= p.idle_after_secs);
        if !idle {
            return Some(Waiting::NotIdle);
        }
    }
    if let (Some(max), Some(t)) = (p.gpu_max_temp_c, r.gpu_temp_c)
        && t >= max
    {
        return Some(Waiting::GpuHot {
            temp_c: t,
            max_c: max,
        });
    }
    if p.yield_to_other_gpu_apps
        && let Some(u) = r.gpu_busy_percent
        && u >= OTHER_GPU_USE_PERCENT
    {
        return Some(Waiting::GpuInUse { percent: u });
    }
    None
}

/// The task the worker computes.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Running {
    pub task_id: String,
    pub kind: String,
    pub started_ms: u64,
}

/// What the worker is doing, for the owner.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Status {
    /// `starting`, `waiting_authorization`, `ready` (asking for work), `computing`, `waiting`
    /// (the policy holds it back: see `waiting`), `stopped`.
    pub state: String,
    pub waiting: Option<Waiting>,
    pub task: Option<Running>,
    pub readings: Readings,
    /// Tasks submitted and abandoned since the worker started.
    pub submitted: u64,
    pub abandoned: u64,
    /// Credits the machine earned (from the server), text in TC.
    pub earned: Option<String>,
    /// Computations of this machine the grid accepted (from the server).
    pub accepted: Option<u64>,
    pub worker_id: Option<String>,
    pub engine: String,
    pub version: String,
    pub policy: Policy,
}
