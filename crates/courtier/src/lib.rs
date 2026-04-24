//! HTTP fetch, authentication, and platform client infrastructure.
//!
//! `courtier` replaces `orkester-io` and adds:
//! - A proper [`FetchError`] type that preserves HTTP status codes
//! - A `request()` method on [`AssetAccessor`] supporting any HTTP verb
//! - Authentication infrastructure ([`auth`], [`authenticated`])
//! - A common [`Client`] trait for service clients
//! - Service clients: [`esri`], [`ion`], [`itwin`]

pub mod auth;
pub mod authenticated;
pub mod client;
pub mod esri;
pub mod fetch;
pub mod gzip;
pub mod ion;
pub mod itwin;
pub mod rest;

#[cfg(not(target_arch = "wasm32"))]
pub mod archive;
#[cfg(not(target_arch = "wasm32"))]
pub mod file;
#[cfg(not(target_arch = "wasm32"))]
pub mod http;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

// Re-export the core types at the crate root for convenience.
pub use auth::{ApiKeyAuth, AuthProvider, BearerTokenAuth};
pub use authenticated::AuthenticatedAccessor;
pub use client::Client;
pub use fetch::{AssetAccessor, AssetResponse, ContentEncoding, FetchError, RequestPriority};
pub use gzip::GunzipAccessor;
pub use rest::{PagedList, fetch_json, parse_json, post_json};

#[cfg(not(target_arch = "wasm32"))]
pub use archive::ArchiveAccessor;
#[cfg(not(target_arch = "wasm32"))]
pub use file::FileAccessor;
#[cfg(not(target_arch = "wasm32"))]
pub use http::{HttpAccessor, RetryConfig};

#[cfg(target_arch = "wasm32")]
pub use wasm::WasmFetchAccessor;
