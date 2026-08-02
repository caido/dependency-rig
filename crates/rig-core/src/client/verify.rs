use crate::{http_client, provider_response, wasm_compat::WasmCompatSend};
use thiserror::Error;

/// Errors from provider client verification.
///
/// Inspect provider failures with [`Self::provider_response_body`],
/// [`Self::provider_response_json`], and [`Self::provider_response_status`].
///
/// Note: no provider path currently constructs [`Self::ProviderResponse`] for
/// verification. Authentication failures surface as [`Self::InvalidAuthentication`]
/// with their original HTTP error attached, while other verify failures surface
/// as [`Self::HttpError`]. The provider response helpers inspect both variants.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum VerifyError {
    #[error("invalid authentication: {0}")]
    InvalidAuthentication(#[source] http_client::Error),
    #[error("provider error: {0}")]
    ProviderError(String),
    /// Raw error response preserved from the provider
    #[error("provider response error: {0}")]
    ProviderResponse(provider_response::ProviderResponseError),
    #[error("http error: {0}")]
    HttpError(
        #[from]
        #[source]
        http_client::Error,
    ),
}

crate::provider_response::impl_provider_response_helpers!(
    VerifyError,
    http_variant = InvalidAuthentication
);

/// A provider client that can verify the configuration.
/// Clone is required for conversions between client types.
pub trait VerifyClient {
    /// Verify the configuration.
    fn verify(&self) -> impl Future<Output = Result<(), VerifyError>> + WasmCompatSend;
}

#[cfg(test)]
mod provider_response_tests {
    use super::*;
    use http::StatusCode;

    #[test]
    fn verify_error_provider_response_helpers_with_preserved_json_body() {
        let body = r#"{"error":{"message":"rate limited"}}"#;
        let error = VerifyError::ProviderResponse(provider_response::ProviderResponseError {
            status: None,
            body: body.to_string(),
        });

        assert_eq!(error.provider_response_body(), Some(body));
        assert_eq!(error.provider_response_status(), None);
        assert_eq!(
            error.provider_response_json().expect("valid JSON"),
            Some(serde_json::json!({ "error": { "message": "rate limited" } }))
        );
    }

    #[test]
    fn verify_error_provider_response_helpers_with_http_non_success() {
        let body = r#"{"error":{"message":"bad request"}}"#;
        let error = VerifyError::HttpError(http_client::Error::InvalidStatusCodeWithMessage(
            StatusCode::BAD_REQUEST,
            body.to_string(),
        ));

        assert_eq!(error.provider_response_body(), Some(body));
        assert_eq!(
            error.provider_response_status(),
            Some(StatusCode::BAD_REQUEST)
        );
        assert_eq!(
            error.provider_response_json().expect("valid JSON"),
            Some(serde_json::json!({ "error": { "message": "bad request" } }))
        );
    }

    #[test]
    fn verify_error_provider_response_helpers_with_preserved_plain_text_body() {
        let error = VerifyError::ProviderResponse(provider_response::ProviderResponseError {
            status: None,
            body: "not json".to_string(),
        });

        assert_eq!(error.provider_response_body(), Some("not json"));
        assert!(error.provider_response_json().is_err());
    }

    #[test]
    fn verify_error_provider_error_is_not_a_provider_response() {
        let error = VerifyError::ProviderError("internal diagnostic".to_string());

        assert_eq!(error.provider_response_body(), None);
        assert_eq!(error.provider_response_status(), None);
        assert_eq!(error.provider_response_json().expect("no body"), None);
    }

    #[test]
    fn verify_error_provider_response_helpers_with_unrelated_variant() {
        let error = VerifyError::ProviderError("invalid authentication".to_string());

        assert_eq!(error.provider_response_body(), None);
        assert_eq!(error.provider_response_status(), None);
        assert_eq!(error.provider_response_json().expect("no body"), None);
    }

    #[tokio::test]
    async fn verify_preserves_status_and_body_on_provider_error_response() {
        use crate::client::VerifyClient;
        use crate::providers::openai::Client;
        use crate::test_utils::RecordingHttpClient;

        let body = r#"{"error":{"message":"server exploded","type":"server_error"}}"#;
        let http_client =
            RecordingHttpClient::with_error_response(StatusCode::INTERNAL_SERVER_ERROR, body);
        let client = Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");

        let error = client
            .verify()
            .await
            .expect_err("verify should fail on a 500 response");

        assert_eq!(
            error.provider_response_status(),
            Some(StatusCode::INTERNAL_SERVER_ERROR)
        );
        assert_eq!(error.provider_response_body(), Some(body));
        let json = error
            .provider_response_json()
            .expect("raw body should be valid JSON")
            .expect("parsed JSON should be present");
        assert_eq!(json["error"]["type"], "server_error");
    }

    #[tokio::test]
    async fn verify_normalizes_only_unauthorized_as_invalid_authentication() {
        use crate::client::VerifyClient;
        use crate::providers::openai::Client;
        use crate::test_utils::RecordingHttpClient;

        let client = Client::builder()
            .api_key("test-key")
            .http_client(RecordingHttpClient::with_error(
                StatusCode::UNAUTHORIZED,
                "invalid API key",
            ))
            .build()
            .expect("build client");

        let error = client
            .verify()
            .await
            .expect_err("verify should reject a 401 response");

        let VerifyError::InvalidAuthentication(source) = &error else {
            panic!("401 should normalize to invalid authentication");
        };
        assert_eq!(source.non_success_status(), Some(StatusCode::UNAUTHORIZED));
        assert_eq!(source.non_success_body(), Some("invalid API key"));
        assert_eq!(
            error.provider_response_status(),
            Some(StatusCode::UNAUTHORIZED)
        );
        assert_eq!(error.provider_response_body(), Some("invalid API key"));
    }

    #[tokio::test]
    async fn verify_preserves_forbidden_status_instead_of_normalizing_authentication() {
        use crate::client::VerifyClient;
        use crate::providers::openai::Client;
        use crate::test_utils::RecordingHttpClient;

        let body = r#"{"error":{"message":"access denied by policy"}}"#;
        let client = Client::builder()
            .api_key("test-key")
            .http_client(RecordingHttpClient::with_error(StatusCode::FORBIDDEN, body))
            .build()
            .expect("build client");

        let error = client
            .verify()
            .await
            .expect_err("verify should reject a 403 response");

        assert!(matches!(error, VerifyError::HttpError(_)));
        assert_eq!(
            error.provider_response_status(),
            Some(StatusCode::FORBIDDEN)
        );
        assert_eq!(error.provider_response_body(), Some(body));
    }
}
