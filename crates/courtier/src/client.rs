//! Common trait implemented by all service clients.

use crate::fetch::AssetAccessor;
use std::sync::Arc;

/// Common interface for all service clients (ESRI, Ion, iTwin, …).
///
/// Each client stores:
/// - `base_url` — root URL for the service
/// - `accessor` — the `AssetAccessor` used for all requests (may be wrapped
///   with `AuthenticatedAccessor` to handle credentials transparently)
///
/// Use the helpers in [`crate::rest`] to make JSON requests against any client.
pub trait Client: Send + Sync + 'static {
    fn base_url(&self) -> &str;
    fn accessor(&self) -> &Arc<dyn AssetAccessor>;
}
