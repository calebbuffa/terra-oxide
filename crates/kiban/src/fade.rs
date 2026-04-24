//! [`FadeStrategy`] - pluggable LOD fade-in / fade-out alpha curves.
//!
//! When LOD transitions are enabled (`StreamingOptions::enable_lod_transition`)
//! the engine blends between the outgoing coarse tile and the incoming fine
//! tile over a configurable period.  The [`FadeStrategy`] trait controls how
//! a progress value in `[0, 1]` is mapped to a visual alpha so applications
//! can substitute ease curves, gamma correction, or application-specific
//! blending without modifying the engine.
//!
//! # Extension
//!
//! ```rust,ignore
//! use kiban::{FadeStrategy, ViewGroup};
//! use std::sync::Arc;
//!
//! pub struct EaseInOutFade;
//! impl FadeStrategy for EaseInOutFade {
//!     fn fade_out_alpha(&self, p: f32) -> f32 {
//!         let t = 1.0 - p;
//!         // smoothstep: 3t^2 − 2t³
//!         t * t * (3.0 - 2.0 * t)
//!     }
//!     fn fade_in_alpha(&self, p: f32) -> f32 {
//!         p * p * (3.0 - 2.0 * p)
//!     }
//! }
//!
//! let vg = ViewGroup::with_fade_strategy(Arc::new(EaseInOutFade));
//! ```

/// Controls the per-frame alpha for tiles entering or leaving the render set
/// during LOD cross-fade transitions.
///
/// # Contract
///
/// - `progress` is always in `[0.0, 1.0]`.
/// - `fade_out_alpha` maps `0.0` (just started fading out) -> `1.0`
///   (fully opaque) and `1.0` (done) -> `0.0` (fully transparent).
/// - `fade_in_alpha` maps `0.0` (just appeared) -> `0.0` (fully transparent)
///   and `1.0` (done) -> `1.0` (fully opaque).
///
/// Both functions must return values in `[0.0, 1.0]`.  Returning a value
/// outside that range will produce visible artefacts (overdraw or no draw).
pub trait FadeStrategy: Send + Sync + 'static {
    /// Alpha for a tile that is fading **out** (being replaced by finer LOD).
    ///
    /// `progress` = 0 -> tile just started fading out (alpha near 1).
    /// `progress` = 1 -> tile is fully transparent (fade complete).
    fn fade_out_alpha(&self, progress: f32) -> f32;

    /// Alpha for a tile that is fading **in** (new fine-LOD tile appearing).
    ///
    /// `progress` = 0 -> tile just appeared (alpha near 0).
    /// `progress` = 1 -> tile is fully opaque (fade complete).
    fn fade_in_alpha(&self, progress: f32) -> f32;
}

/// Linear fade - the default.
///
/// `fade_out_alpha(p) = 1 − p`
/// `fade_in_alpha(p)  = p`
///
/// This matches the original behaviour that was previously hardcoded in
/// `ViewGroup::commit_result`.
pub struct LinearFadeStrategy;

impl FadeStrategy for LinearFadeStrategy {
    #[inline]
    fn fade_out_alpha(&self, progress: f32) -> f32 {
        (1.0 - progress).max(0.0)
    }

    #[inline]
    fn fade_in_alpha(&self, progress: f32) -> f32 {
        progress.min(1.0)
    }
}
