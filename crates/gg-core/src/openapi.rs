//! OpenAPI schema descriptions of domain scalars (feature `openapi`).

use utoipa::openapi::RefOr;
use utoipa::openapi::schema::{ObjectBuilder, Schema, SchemaType, Type};
use utoipa::{PartialSchema, ToSchema};

macro_rules! string_schema {
    ($t:ty, $desc:literal) => {
        impl PartialSchema for $t {
            fn schema() -> RefOr<Schema> {
                ObjectBuilder::new()
                    .schema_type(SchemaType::Type(Type::String))
                    .description(Some($desc))
                    .into()
            }
        }
        impl ToSchema for $t {}
    };
}

macro_rules! enum_schema {
    ($t:ty) => {
        impl PartialSchema for $t {
            fn schema() -> RefOr<Schema> {
                ObjectBuilder::new()
                    .schema_type(SchemaType::Type(Type::String))
                    .enum_values(Some(<$t>::ALL.iter().map(|v| v.as_str())))
                    .into()
            }
        }
        impl ToSchema for $t {}
    };
}

string_schema!(crate::ids::ResearcherId, "UUID");
string_schema!(crate::ids::ProjectId, "UUID");
string_schema!(crate::ids::ExperimentId, "UUID");
string_schema!(crate::ids::JobId, "UUID");
string_schema!(crate::ids::TaskId, "UUID");
string_schema!(crate::ids::AttemptId, "UUID");
string_schema!(crate::ids::LeaseId, "UUID");
string_schema!(crate::ids::WorkerId, "UUID");
string_schema!(crate::ids::ValidatorId, "UUID");
string_schema!(crate::ids::NodeId, "UUID");
string_schema!(crate::ids::ReaderId, "UUID");
string_schema!(crate::ids::SubmissionId, "UUID");
string_schema!(crate::ids::DecisionId, "UUID");
string_schema!(crate::ids::ReceiptId, "UUID");
string_schema!(crate::ids::LedgerTxnId, "UUID");
string_schema!(crate::ids::ReservationId, "UUID");
string_schema!(crate::ids::AuditEventId, "UUID");
string_schema!(
    crate::digest::Sha256Digest,
    "SHA-256 digest, 64 lowercase hex characters"
);
string_schema!(
    crate::credits::Credits,
    "Integer amount of test credits (TC) as a decimal string"
);
string_schema!(
    crate::network::NetworkId,
    "Network identifier [a-z0-9-]{1,32}"
);

impl PartialSchema for crate::time::UnixMillis {
    fn schema() -> RefOr<Schema> {
        ObjectBuilder::new()
            .schema_type(SchemaType::Type(Type::Integer))
            .description(Some("Milliseconds since the Unix epoch (UTC)"))
            .into()
    }
}
impl ToSchema for crate::time::UnixMillis {}

enum_schema!(crate::state::TaskState);
enum_schema!(crate::state::AttemptState);
enum_schema!(crate::state::AttemptRole);
enum_schema!(crate::state::VerificationOutcome);
enum_schema!(crate::state::VerificationState);
enum_schema!(crate::state::RewardState);
enum_schema!(crate::state::JobState);
enum_schema!(crate::state::AdmissionState);
enum_schema!(crate::state::ProjectState);
enum_schema!(crate::state::ProjectRole);
