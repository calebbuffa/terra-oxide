//! Culling result for visibility tests.

/// Result of testing a bounding volume against a culling volume or plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CullingResult {
    /// Entirely inside the culling volume.
    Inside,
    /// Entirely outside the culling volume.
    #[default]
    Outside,
    /// Partially inside and partially outside.
    Intersecting,
}

impl CullingResult {
    /// Returns `true` if the volume is not entirely outside (i.e., `Inside` or `Intersecting`).
    #[inline]
    pub fn is_visible(self) -> bool {
        self != Self::Outside
    }
}
