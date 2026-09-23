//! State machines for tasks, attempts, verifications and rewards.
//!
//! These functions are the executable source of truth for `gg-server/docs/SPEC.md` §3. Handlers in the
//! control plane must call them and persist the result; they never invent transitions.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

macro_rules! text_enum {
    ($(#[$meta:meta])* $name:ident { $($variant:ident => $text:literal),+ $(,)? }) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
        #[serde(rename_all = "snake_case")]
        #[allow(missing_docs)]
        pub enum $name { $($variant),+ }

        impl $name {
            /// Stable textual form used in the database and the API.
            pub const fn as_str(self) -> &'static str {
                match self { $(Self::$variant => $text),+ }
            }
            /// All variants.
            pub const ALL: &'static [Self] = &[$(Self::$variant),+];
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str(self.as_str()) }
        }

        impl FromStr for $name {
            type Err = UnknownState;
            fn from_str(s: &str) -> Result<Self, Self::Err> {
                match s { $($text => Ok(Self::$variant),)+ _ => Err(UnknownState(s.to_owned())) }
            }
        }
    };
}

/// Unknown textual state.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[error("unknown state: {0}")]
pub struct UnknownState(pub String);

text_enum!(
    /// State of a task instance.
    TaskState {
        Queued => "queued",
        Assigned => "assigned",
        Submitted => "submitted",
        Accepted => "accepted",
        Rejected => "rejected",
        Inconclusive => "inconclusive",
        Disputed => "disputed",
        Failed => "failed",
        Cancelled => "cancelled",
    }
);

text_enum!(
    /// State of an attempt.
    AttemptState {
        Active => "active",
        Submitted => "submitted",
        Expired => "expired",
        Abandoned => "abandoned",
        Superseded => "superseded",
        Accepted => "accepted",
        Rejected => "rejected",
        Inconclusive => "inconclusive",
    }
);

text_enum!(
    /// Role of an attempt. A contribution is an accepted attempt for an authorized role.
    AttemptRole {
        Primary => "primary",
        Replica => "replica",
        Audit => "audit",
    }
);

text_enum!(
    /// Outcome of a verification decision.
    VerificationOutcome {
        Accepted => "accepted",
        Rejected => "rejected",
        Inconclusive => "inconclusive",
    }
);

text_enum!(
    /// State of a verification work item.
    VerificationState {
        Pending => "pending",
        InProgress => "in_progress",
        Decided => "decided",
    }
);

text_enum!(
    /// Reward (payment) state, separate from the scientific state.
    RewardState {
        None => "none",
        Earned => "earned",
        Batched => "batched",
        Settled => "settled",
    }
);

text_enum!(
    /// Derived job state.
    JobState {
        Queued => "queued",
        Running => "running",
        Succeeded => "succeeded",
        Failed => "failed",
        Cancelled => "cancelled",
    }
);

text_enum!(
    /// Researcher admission state.
    AdmissionState {
        Pending => "pending",
        Approved => "approved",
        Suspended => "suspended",
        Rejected => "rejected",
    }
);

text_enum!(
    /// Project state.
    ProjectState {
        Active => "active",
        Suspended => "suspended",
        Closed => "closed",
    }
);

text_enum!(
    /// Membership role in a project.
    ProjectRole {
        Owner => "owner",
        Member => "member",
        Viewer => "viewer",
    }
);

/// Who is requesting a transition. Authorization is checked before, but the state machine also
/// refuses transitions that no actor of that kind may trigger.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Authority {
    /// The control server itself (scheduler, reaper, decision application).
    Server,
    /// A worker holding a valid lease for the attempt in question.
    Worker,
    /// An administered validator.
    Validator,
    /// A researcher with owner/member role on the project.
    Researcher,
    /// An operator/admin.
    Admin,
}

/// Events that drive the task state machine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TaskEvent {
    /// Scheduler assigned an attempt (project active, kind allowed, budget reserved).
    Assign,
    /// Worker submitted a result on the current generation with a valid lease.
    Submit,
    /// Lease lost (expired/abandoned). `attempts_left` tells whether re-queueing is allowed.
    LeaseLost {
        /// Whether another attempt may be scheduled.
        attempts_left: bool,
    },
    /// Validator decided.
    Decide(VerificationOutcome),
    /// Server retry policy after a non-accepted decision.
    Retry {
        /// Whether another attempt may be scheduled.
        attempts_left: bool,
    },
    /// Decisions on the same submission disagreed: the task waits for a human resolution.
    Dispute,
    /// An operator resolved a dispute with a final outcome.
    Resolve(VerificationOutcome),
    /// Cancellation.
    Cancel,
}

/// Invalid transition.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum TransitionError {
    /// The event is not allowed from this state.
    #[error("event {event:?} not allowed from state {from}")]
    NotAllowed {
        /// Current state.
        from: &'static str,
        /// Event.
        event: String,
    },
    /// The authority is not allowed to trigger this event.
    #[error("authority {authority:?} may not trigger {event:?}")]
    Unauthorized {
        /// Authority.
        authority: Authority,
        /// Event.
        event: String,
    },
}

impl TaskState {
    /// Terminal states never change again.
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Accepted | Self::Failed | Self::Cancelled)
    }

    /// Whether the scheduler may assign this task.
    pub const fn is_assignable(self) -> bool {
        matches!(self, Self::Queued)
    }

    /// Compute the next state for `event` requested by `authority`.
    pub fn next(
        self,
        event: TaskEvent,
        authority: Authority,
    ) -> Result<TaskState, TransitionError> {
        use Authority as A;
        use TaskEvent as E;
        use TaskState as S;

        let allowed_authority = match event {
            E::Assign | E::LeaseLost { .. } | E::Retry { .. } | E::Dispute => {
                matches!(authority, A::Server)
            }
            E::Resolve(_) => matches!(authority, A::Admin),
            E::Submit => matches!(authority, A::Worker),
            E::Decide(_) => matches!(authority, A::Validator | A::Server),
            E::Cancel => matches!(authority, A::Researcher | A::Admin | A::Server),
        };
        if !allowed_authority {
            return Err(TransitionError::Unauthorized {
                authority,
                event: format!("{event:?}"),
            });
        }

        let next = match (self, event) {
            (S::Queued, E::Assign) => S::Assigned,
            (S::Assigned, E::Submit) => S::Submitted,
            (
                S::Assigned,
                E::LeaseLost {
                    attempts_left: true,
                },
            ) => S::Queued,
            (
                S::Assigned,
                E::LeaseLost {
                    attempts_left: false,
                },
            ) => S::Failed,
            (S::Submitted, E::Dispute) => S::Disputed,
            (S::Disputed, E::Resolve(VerificationOutcome::Accepted)) => S::Accepted,
            (S::Disputed, E::Resolve(VerificationOutcome::Rejected)) => S::Rejected,
            (S::Disputed, E::Resolve(VerificationOutcome::Inconclusive)) => S::Inconclusive,
            (S::Submitted, E::Decide(VerificationOutcome::Accepted)) => S::Accepted,
            (S::Submitted, E::Decide(VerificationOutcome::Rejected)) => S::Rejected,
            (S::Submitted, E::Decide(VerificationOutcome::Inconclusive)) => S::Inconclusive,
            (
                S::Rejected | S::Inconclusive,
                E::Retry {
                    attempts_left: true,
                },
            ) => S::Queued,
            (
                S::Rejected | S::Inconclusive,
                E::Retry {
                    attempts_left: false,
                },
            ) => S::Failed,
            (
                S::Queued
                | S::Assigned
                | S::Submitted
                | S::Rejected
                | S::Inconclusive
                | S::Disputed,
                E::Cancel,
            ) => S::Cancelled,
            (from, event) => {
                return Err(TransitionError::NotAllowed {
                    from: from.as_str(),
                    event: format!("{event:?}"),
                });
            }
        };
        Ok(next)
    }
}

/// Events for the attempt state machine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttemptEvent {
    /// Result submitted with valid lease.
    Submit,
    /// Lease expired (server clock).
    Expire,
    /// Worker abandoned the attempt.
    Abandon,
    /// Task canceled (or reassigned) while this attempt existed.
    Supersede,
    /// Decision applied.
    Decide(VerificationOutcome),
}

impl AttemptState {
    /// Terminal attempt states.
    pub const fn is_terminal(self) -> bool {
        !matches!(self, Self::Active | Self::Submitted)
    }

    /// Whether a result may still be submitted on this attempt (subject to lease and generation).
    pub const fn accepts_submission(self) -> bool {
        matches!(self, Self::Active)
    }

    /// Next attempt state.
    pub fn next(self, event: AttemptEvent) -> Result<AttemptState, TransitionError> {
        use AttemptEvent as E;
        use AttemptState as S;
        let next = match (self, event) {
            (S::Active, E::Submit) => S::Submitted,
            (S::Active, E::Expire) => S::Expired,
            (S::Active, E::Abandon) => S::Abandoned,
            (S::Active | S::Submitted, E::Supersede) => S::Superseded,
            (S::Submitted, E::Decide(VerificationOutcome::Accepted)) => S::Accepted,
            (S::Submitted, E::Decide(VerificationOutcome::Rejected)) => S::Rejected,
            (S::Submitted, E::Decide(VerificationOutcome::Inconclusive)) => S::Inconclusive,
            (from, event) => {
                return Err(TransitionError::NotAllowed {
                    from: from.as_str(),
                    event: format!("{event:?}"),
                });
            }
        };
        Ok(next)
    }
}

impl JobState {
    /// Derive a job state from its task states. An empty job is `queued`.
    pub fn derive(tasks: &[TaskState], cancelled: bool) -> JobState {
        if cancelled {
            return JobState::Cancelled;
        }
        if tasks.contains(&TaskState::Failed) {
            return JobState::Failed;
        }
        if !tasks.is_empty() && tasks.iter().all(|t| *t == TaskState::Accepted) {
            return JobState::Succeeded;
        }
        if tasks.iter().any(|t| {
            matches!(
                t,
                TaskState::Assigned
                    | TaskState::Submitted
                    | TaskState::Accepted
                    | TaskState::Disputed
            )
        }) {
            return JobState::Running;
        }
        JobState::Queued
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn happy_path() {
        let s = TaskState::Queued;
        let s = s.next(TaskEvent::Assign, Authority::Server).unwrap();
        assert_eq!(s, TaskState::Assigned);
        let s = s.next(TaskEvent::Submit, Authority::Worker).unwrap();
        assert_eq!(s, TaskState::Submitted);
        let s = s
            .next(
                TaskEvent::Decide(VerificationOutcome::Accepted),
                Authority::Validator,
            )
            .unwrap();
        assert_eq!(s, TaskState::Accepted);
        assert!(s.is_terminal());
    }

    #[test]
    fn cancelled_never_assigned_again() {
        let s = TaskState::Queued
            .next(TaskEvent::Cancel, Authority::Researcher)
            .unwrap();
        assert_eq!(s, TaskState::Cancelled);
        assert!(s.next(TaskEvent::Assign, Authority::Server).is_err());
        assert!(s.next(TaskEvent::Submit, Authority::Worker).is_err());
        assert!(!s.is_assignable());
    }

    #[test]
    fn accepted_is_final_even_for_admin() {
        let s = TaskState::Accepted;
        assert!(s.next(TaskEvent::Cancel, Authority::Admin).is_err());
        assert!(
            s.next(
                TaskEvent::Retry {
                    attempts_left: true
                },
                Authority::Server
            )
            .is_err()
        );
    }

    #[test]
    fn authority_is_enforced() {
        assert_eq!(
            TaskState::Queued.next(TaskEvent::Assign, Authority::Worker),
            Err(TransitionError::Unauthorized {
                authority: Authority::Worker,
                event: "Assign".into()
            })
        );
        assert!(
            TaskState::Assigned
                .next(TaskEvent::Submit, Authority::Researcher)
                .is_err()
        );
        assert!(
            TaskState::Submitted
                .next(
                    TaskEvent::Decide(VerificationOutcome::Accepted),
                    Authority::Worker
                )
                .is_err()
        );
    }

    #[test]
    fn lease_lost_requeues_or_fails() {
        assert_eq!(
            TaskState::Assigned
                .next(
                    TaskEvent::LeaseLost {
                        attempts_left: true
                    },
                    Authority::Server
                )
                .unwrap(),
            TaskState::Queued
        );
        assert_eq!(
            TaskState::Assigned
                .next(
                    TaskEvent::LeaseLost {
                        attempts_left: false
                    },
                    Authority::Server
                )
                .unwrap(),
            TaskState::Failed
        );
    }

    #[test]
    fn inconclusive_is_never_promoted_to_accepted() {
        let s = TaskState::Inconclusive;
        for ev in [
            TaskEvent::Assign,
            TaskEvent::Submit,
            TaskEvent::Decide(VerificationOutcome::Accepted),
        ] {
            for auth in [
                Authority::Server,
                Authority::Worker,
                Authority::Validator,
                Authority::Admin,
            ] {
                assert_ne!(s.next(ev, auth).ok(), Some(TaskState::Accepted));
            }
        }
    }

    #[test]
    fn attempt_machine() {
        assert_eq!(
            AttemptState::Active.next(AttemptEvent::Submit).unwrap(),
            AttemptState::Submitted
        );
        assert!(AttemptState::Expired.next(AttemptEvent::Submit).is_err());
        assert!(
            AttemptState::Superseded
                .next(AttemptEvent::Decide(VerificationOutcome::Accepted))
                .is_err()
        );
        assert!(!AttemptState::Expired.accepts_submission());
    }

    #[test]
    fn job_derivation() {
        use TaskState as T;
        assert_eq!(JobState::derive(&[], false), JobState::Queued);
        assert_eq!(
            JobState::derive(&[T::Queued, T::Queued], false),
            JobState::Queued
        );
        assert_eq!(
            JobState::derive(&[T::Accepted, T::Queued], false),
            JobState::Running
        );
        assert_eq!(
            JobState::derive(&[T::Accepted, T::Accepted], false),
            JobState::Succeeded
        );
        assert_eq!(
            JobState::derive(&[T::Accepted, T::Failed], false),
            JobState::Failed
        );
        assert_eq!(JobState::derive(&[T::Accepted], true), JobState::Cancelled);
    }

    #[test]
    fn text_roundtrip() {
        for s in TaskState::ALL {
            assert_eq!(s.as_str().parse::<TaskState>().unwrap(), *s);
        }
        assert_eq!(
            serde_json::to_string(&TaskState::Queued).unwrap(),
            "\"queued\""
        );
        assert_eq!(
            serde_json::to_string(&VerificationState::InProgress).unwrap(),
            "\"in_progress\""
        );
    }

    fn any_event() -> impl Strategy<Value = TaskEvent> {
        prop_oneof![
            Just(TaskEvent::Assign),
            Just(TaskEvent::Submit),
            any::<bool>().prop_map(|b| TaskEvent::LeaseLost { attempts_left: b }),
            Just(TaskEvent::Decide(VerificationOutcome::Accepted)),
            Just(TaskEvent::Decide(VerificationOutcome::Rejected)),
            Just(TaskEvent::Decide(VerificationOutcome::Inconclusive)),
            any::<bool>().prop_map(|b| TaskEvent::Retry { attempts_left: b }),
            Just(TaskEvent::Cancel),
        ]
    }

    fn any_authority() -> impl Strategy<Value = Authority> {
        prop_oneof![
            Just(Authority::Server),
            Just(Authority::Worker),
            Just(Authority::Validator),
            Just(Authority::Researcher),
            Just(Authority::Admin),
        ]
    }

    proptest! {
        /// Terminal states are absorbing and only the validator/server path can reach `accepted`.
        #[test]
        fn terminal_states_are_absorbing(events in proptest::collection::vec((any_event(), any_authority()), 0..20)) {
            let mut s = TaskState::Queued;
            let mut reached_terminal_at: Option<TaskState> = None;
            for (ev, auth) in events {
                if let Ok(n) = s.next(ev, auth) {
                    prop_assert!(reached_terminal_at.is_none(), "terminal state changed");
                    if n == TaskState::Accepted {
                        prop_assert_eq!(s, TaskState::Submitted);
                        prop_assert!(matches!(ev, TaskEvent::Decide(VerificationOutcome::Accepted)));
                        prop_assert!(matches!(auth, Authority::Validator | Authority::Server));
                    }
                    s = n;
                    if s.is_terminal() { reached_terminal_at = Some(s); }
                }
            }
        }
    }
}
