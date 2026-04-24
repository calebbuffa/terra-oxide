/// Returns the current time in milliseconds since the Unix epoch.
#[cfg(target_arch = "wasm32")]
pub fn time_now_ms() -> u64 {
    js_sys::Date::now() as u64
}

/// Returns the current time in milliseconds since the Unix epoch.
#[cfg(not(target_arch = "wasm32"))]
pub fn time_now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Returns the current time in whole seconds since the Unix epoch.
pub fn time_now_secs() -> u64 {
    time_now_ms() / 1000
}
