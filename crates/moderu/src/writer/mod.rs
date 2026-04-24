mod error;
mod glb;
mod writer;

pub use error::{WriteError, WriteResult};
pub use writer::{GltfWriter, GltfWriterOptions};

/// A convenience type alias for `Result<T, WriteError>`.
pub type Result<T> = std::result::Result<T, WriteError>;
