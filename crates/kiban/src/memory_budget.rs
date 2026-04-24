//! Memory budget tracking and management.
//!
//! Provides a high-level API for monitoring and controlling tile cache memory usage.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Current memory pressure state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryPressure {
    /// Memory usage is within budget.
    Nominal,
    /// Memory usage is between budget and budget+overflow.
    Elevated,
    /// Memory usage exceeds budget+overflow.
    Critical,
}

impl MemoryPressure {
    /// Calculate pressure from usage and budget.
    pub fn from_usage(usage_bytes: usize, budget_bytes: usize, overflow_bytes: usize) -> Self {
        let max = budget_bytes.saturating_add(overflow_bytes);
        if usage_bytes > max {
            MemoryPressure::Critical
        } else if usage_bytes > budget_bytes {
            MemoryPressure::Elevated
        } else {
            MemoryPressure::Nominal
        }
    }

    /// Human-readable description.
    pub fn description(&self) -> &'static str {
        match self {
            MemoryPressure::Nominal => "Memory usage is nominal",
            MemoryPressure::Elevated => "Memory usage is elevated (approaching budget)",
            MemoryPressure::Critical => "Memory usage is critical (exceeds budget)",
        }
    }
}

/// Callback fired when memory pressure changes.
pub type MemoryPressureCallback = Box<dyn Fn(MemoryPressure, MemoryUsage) + Send + Sync + 'static>;

/// Snapshot of current memory usage.
#[derive(Debug, Clone, Copy)]
pub struct MemoryUsage {
    /// Total bytes of loaded tile content.
    pub used_bytes: usize,
    /// Maximum budget before eviction pressure increases.
    pub budget_bytes: usize,
    /// Additional buffer before critical pressure.
    pub overflow_bytes: usize,
    /// Current pressure state.
    pub pressure: MemoryPressure,
}

impl MemoryUsage {
    /// Memory percentage (0-100, can exceed 100 if over budget).
    pub fn usage_percent(&self) -> f32 {
        let total = self.budget_bytes.saturating_add(self.overflow_bytes);
        if total == 0 {
            0.0
        } else {
            (self.used_bytes as f32 / total as f32) * 100.0
        }
    }

    /// Bytes available before reaching budget.
    pub fn available_bytes(&self) -> usize {
        self.budget_bytes.saturating_sub(self.used_bytes)
    }

    /// Bytes over budget (0 if under budget).
    pub fn overflow_bytes_used(&self) -> usize {
        self.used_bytes.saturating_sub(self.budget_bytes)
    }
}

/// Memory budget configuration and tracking.
pub struct MemoryBudget {
    /// Maximum bytes to cache before memory pressure eviction starts.
    pub max_bytes: usize,

    /// Additional overflow buffer before critical pressure.
    /// (Total threshold = max_bytes + overflow_bytes)
    pub overflow_bytes: usize,

    /// Callback fired when pressure state changes (Nominal -> Elevated, etc.)
    pub on_pressure_change: Option<MemoryPressureCallback>,

    /// Callback fired every frame with current usage (for metrics/logging).
    pub on_usage_update: Option<Box<dyn Fn(MemoryUsage) + Send + Sync + 'static>>,

    /// Current tracked memory usage (atomic for lock-free reads).
    current_usage_bytes: Arc<AtomicUsize>,

    /// Last reported pressure state (to detect changes).
    last_pressure: std::sync::Mutex<MemoryPressure>,
}

impl MemoryBudget {
    /// Create a new memory budget with given limits.
    ///
    /// # Arguments
    /// * `max_bytes` - Target cache size before eviction pressure increases
    /// * `overflow_bytes` - Extra buffer before critical pressure (default: 10% of max)
    pub fn new(max_bytes: usize) -> Self {
        let overflow_bytes = (max_bytes as f64 * 0.1) as usize;
        Self {
            max_bytes,
            overflow_bytes,
            on_pressure_change: None,
            on_usage_update: None,
            current_usage_bytes: Arc::new(AtomicUsize::new(0)),
            last_pressure: std::sync::Mutex::new(MemoryPressure::Nominal),
        }
    }

    /// Set a callback fired when memory pressure changes.
    pub fn on_pressure_change<F>(mut self, callback: F) -> Self
    where
        F: Fn(MemoryPressure, MemoryUsage) + Send + Sync + 'static,
    {
        self.on_pressure_change = Some(Box::new(callback));
        self
    }

    /// Set a callback fired every frame with current memory usage.
    pub fn on_usage_update<F>(mut self, callback: F) -> Self
    where
        F: Fn(MemoryUsage) + Send + Sync + 'static,
    {
        self.on_usage_update = Some(Box::new(callback));
        self
    }

    /// Update current memory usage (called by Layer internally).
    pub fn update_usage(&self, bytes: usize) {
        self.current_usage_bytes.store(bytes, Ordering::Release);

        let usage = self.current_usage();

        // Check if pressure state changed
        if let Ok(mut last_pressure) = self.last_pressure.lock() {
            if usage.pressure != *last_pressure {
                *last_pressure = usage.pressure;
                if let Some(callback) = &self.on_pressure_change {
                    callback(usage.pressure, usage);
                }
            }
        }

        // Call update callback if configured
        if let Some(callback) = &self.on_usage_update {
            callback(usage);
        }
    }

    /// Get current memory usage snapshot.
    pub fn current_usage(&self) -> MemoryUsage {
        let used_bytes = self.current_usage_bytes.load(Ordering::Acquire);
        let pressure = MemoryPressure::from_usage(used_bytes, self.max_bytes, self.overflow_bytes);

        MemoryUsage {
            used_bytes,
            budget_bytes: self.max_bytes,
            overflow_bytes: self.overflow_bytes,
            pressure,
        }
    }

    /// Check if memory is under budget.
    pub fn is_under_budget(&self) -> bool {
        self.current_usage_bytes.load(Ordering::Acquire) <= self.max_bytes
    }

    /// Get available bytes before budget threshold.
    pub fn available_bytes(&self) -> usize {
        self.max_bytes
            .saturating_sub(self.current_usage_bytes.load(Ordering::Acquire))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pressure_calculation() {
        assert_eq!(
            MemoryPressure::from_usage(50, 100, 20),
            MemoryPressure::Nominal
        );
        assert_eq!(
            MemoryPressure::from_usage(110, 100, 20),
            MemoryPressure::Elevated
        );
        assert_eq!(
            MemoryPressure::from_usage(150, 100, 20),
            MemoryPressure::Critical
        );
    }

    #[test]
    fn test_memory_budget_creation() {
        let budget = MemoryBudget::new(1024 * 1024);
        assert_eq!(budget.max_bytes, 1024 * 1024);
        assert!(budget.overflow_bytes > 0);
        assert!(budget.is_under_budget());
    }

    #[test]
    fn test_memory_usage_snapshot() {
        let budget = MemoryBudget::new(1000);
        budget.update_usage(500);

        let usage = budget.current_usage();
        assert_eq!(usage.used_bytes, 500);
        assert_eq!(usage.available_bytes(), 500);
        assert_eq!(usage.pressure, MemoryPressure::Nominal);
    }

    #[test]
    fn test_pressure_change_callback() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering as AOrdering};

        let callback_fired = Arc::new(AtomicBool::new(false));
        let fired_clone = Arc::clone(&callback_fired);

        let budget = MemoryBudget::new(100).on_pressure_change(move |_p, _u| {
            fired_clone.store(true, AOrdering::Release);
        });

        // First update: Nominal
        budget.update_usage(50);
        assert!(!callback_fired.load(AOrdering::Acquire));

        // Second update: Elevated (callback should fire)
        budget.update_usage(110);
        assert!(callback_fired.load(AOrdering::Acquire));
    }
}
