//! Typed identifiers. All ids are UUIDs (v7 when generated locally, time-sortable).

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

macro_rules! typed_id {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(pub Uuid);

        impl $name {
            /// Generate a fresh time-ordered identifier.
            pub fn new() -> Self {
                Self(Uuid::now_v7())
            }

            /// Underlying UUID.
            pub fn as_uuid(&self) -> Uuid {
                self.0
            }

            /// Nil identifier, only for tests and fixtures.
            pub const fn nil() -> Self {
                Self(Uuid::nil())
            }

            /// Build from raw 128-bit value (fixtures).
            pub const fn from_u128(v: u128) -> Self {
                Self(Uuid::from_u128(v))
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.0.hyphenated())
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}({})", stringify!($name), self.0.hyphenated())
            }
        }

        impl FromStr for $name {
            type Err = uuid::Error;
            fn from_str(s: &str) -> Result<Self, Self::Err> {
                Ok(Self(Uuid::parse_str(s)?))
            }
        }

        impl From<Uuid> for $name {
            fn from(u: Uuid) -> Self {
                Self(u)
            }
        }

        impl From<$name> for Uuid {
            fn from(id: $name) -> Uuid {
                id.0
            }
        }
    };
}

typed_id!(
    /// A researcher (human identity, authenticated separately from authorization).
    ResearcherId
);
typed_id!(
    /// A project: authorization and budget container.
    ProjectId
);
typed_id!(
    /// An experiment grouping jobs.
    ExperimentId
);
typed_id!(
    /// A job submitted by a researcher.
    JobId
);
typed_id!(
    /// A task instance (distinct from the specification hash).
    TaskId
);
typed_id!(
    /// An attempt of a task assigned to a worker.
    AttemptId
);
typed_id!(
    /// A lease granted for an attempt.
    LeaseId
);
typed_id!(
    /// A worker (contributor agent) identity record.
    WorkerId
);
typed_id!(
    /// A validator identity record.
    ValidatorId
);
typed_id!(
    /// A result submission.
    SubmissionId
);
typed_id!(
    /// A verification decision.
    DecisionId
);
typed_id!(
    /// A contribution receipt; also the global reward identifier.
    ReceiptId
);
typed_id!(
    /// A ledger transaction.
    LedgerTxnId
);
typed_id!(
    /// A budget reservation.
    ReservationId
);
typed_id!(
    /// An audit event.
    AuditEventId
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_roundtrip_text_and_json() {
        let id = TaskId::new();
        let s = id.to_string();
        assert_eq!(TaskId::from_str(&s).unwrap(), id);
        let j = serde_json::to_string(&id).unwrap();
        assert_eq!(j, format!("\"{s}\""));
        assert_eq!(serde_json::from_str::<TaskId>(&j).unwrap(), id);
    }

    #[test]
    fn v7_ids_are_time_ordered() {
        let a = JobId::new();
        let b = JobId::new();
        assert!(a <= b);
    }
}
