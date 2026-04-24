//! URI and file-path utilities.

use std::fmt;

/// An owned, resolved URI string.
///
/// Wraps a `String` but makes the intent explicit and provides common
/// trait implementations for use as map keys, debug output, etc.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Uri(pub(crate) String);

impl Uri {
    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn resolve(&self, key: &str) -> String {
        resolve_url(&self.0, key)
    }

    pub fn extension(&self) -> Option<&str> {
        file_extension(&self.0)
    }
}

impl fmt::Display for Uri {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for Uri {
    fn from(s: String) -> Self {
        Uri(s)
    }
}

impl From<&str> for Uri {
    fn from(s: &str) -> Self {
        Uri(s.to_owned())
    }
}

impl AsRef<str> for Uri {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

/// Resolve `key` relative to `base`.
///
/// If `key` is already absolute (starts with a scheme such as `https://` or
/// `file://`, or with `/` or `\`) it is returned as-is. Otherwise it is
/// appended to the directory portion of `base` (everything up to and including
/// the last `/` or `\`, with query and fragment stripped).
///
/// Handles both Unix `/` and Windows `\` path separators.
///
/// # Examples
///
/// ```
/// use outil::resolve_url;
///
/// assert_eq!(
///     resolve_url("https://example.com/tiles/model.gltf", "buffer0.bin"),
///     "https://example.com/tiles/buffer0.bin"
/// );
/// assert_eq!(
///     resolve_url(r"C:\tiles\model.gltf", "tex.png"),
///     r"C:\tiles\tex.png"
/// );
/// assert_eq!(
///     resolve_url("https://example.com/a.gltf", "/root/b.png"),
///     "/root/b.png"
/// );
/// ```
pub fn resolve_url(base: &str, key: &str) -> String {
    // Already absolute: has a scheme (`://` that appears *before* any `?`/`#`),
    // starts with /, \, or is a Windows drive-letter path (e.g. C:\).
    //
    // The "before `?`/`#`" check matters for keys like `a?next=https://x` -
    // the embedded `://` lives inside the query string and does not make
    // the whole thing absolute.
    let scheme_boundary = key.find("://");
    let query_boundary = key.find(['?', '#']);
    let has_scheme = match (scheme_boundary, query_boundary) {
        (Some(s), Some(q)) => s < q,
        (Some(_), None) => true,
        _ => false,
    };
    let is_absolute = has_scheme
        || key.starts_with('/')
        || key.starts_with('\\')
        || (key.len() >= 2 && key.as_bytes()[1] == b':' && key.as_bytes()[0].is_ascii_alphabetic());
    if is_absolute {
        return key.to_owned();
    }
    let base_path = base.split('?').next().unwrap_or(base);
    let base_path = base_path.split('#').next().unwrap_or(base_path);
    let last_sep = base_path.rfind(|c| c == '/' || c == '\\');
    let dir = last_sep.map_or("", |i| &base_path[..=i]);
    let joined = format!("{dir}{key}");
    // RFC 3986 § 5.2.4 dot-segment removal, applied to the path portion only.
    // Windows paths that contain `\` separators are left untouched to avoid
    // mangling drive-letter prefixes and UNC paths.
    if joined.contains('\\') {
        joined
    } else {
        remove_dot_segments_in_url(&joined)
    }
}

/// Remove `.` and `..` dot-segments from the path portion of a URL-ish string.
///
/// Preserves scheme, authority, query, and fragment. Matches RFC 3986 § 5.2.4
/// closely enough for the paths seen in 3D Tiles / glTF manifests.
fn remove_dot_segments_in_url(url: &str) -> String {
    // Split into (prefix, path, suffix) where:
    //   prefix = everything up to and including the authority's trailing `/`
    //   path   = the path before the query/fragment
    //   suffix = "?query" and/or "#fragment", preserved verbatim
    let (head, suffix) = match url.find(['?', '#']) {
        Some(i) => url.split_at(i),
        None => (url, ""),
    };

    // Identify the boundary between the scheme+authority and the path.
    // For `scheme://host/path/...` the path starts at the 3rd `/`.
    // For `/rooted/path/...`        the path starts at index 0.
    // For `bare/relative/path`      the path starts at index 0.
    let path_start = if let Some(scheme_end) = head.find("://") {
        let authority_start = scheme_end + 3;
        head[authority_start..]
            .find('/')
            .map(|i| authority_start + i)
            .unwrap_or(head.len())
    } else {
        0
    };

    let (prefix, path) = head.split_at(path_start);
    let normalized = remove_dot_segments(path);
    let mut out = String::with_capacity(prefix.len() + normalized.len() + suffix.len());
    out.push_str(prefix);
    out.push_str(&normalized);
    out.push_str(suffix);
    out
}

/// Apply RFC 3986 § 5.2.4 dot-segment removal to a path string.
fn remove_dot_segments(path: &str) -> String {
    // Work segment-by-segment, preserving a leading `/` when present.
    let absolute = path.starts_with('/');
    let trailing_slash = path.ends_with('/') && !path.is_empty();
    let mut out: Vec<&str> = Vec::new();
    for seg in path.split('/') {
        match seg {
            "" | "." => continue,
            ".." => {
                // Pop one real segment. For absolute paths, `..` at the root
                // is discarded (can't go above root). For relative paths, it
                // is preserved when we have nothing to pop, which matches how
                // most HTTP clients and ada-url behave.
                if out.last().is_some_and(|s| *s != "..") {
                    out.pop();
                } else if !absolute {
                    out.push("..");
                }
            }
            other => out.push(other),
        }
    }
    let mut result = String::with_capacity(path.len());
    if absolute {
        result.push('/');
    }
    for (i, seg) in out.iter().enumerate() {
        if i > 0 {
            result.push('/');
        }
        result.push_str(seg);
    }
    if trailing_slash && !result.ends_with('/') {
        result.push('/');
    }
    result
}

/// Extract the file extension from a URL or file path.
///
/// Strips any query string (`?…`) and fragment (`#…`) before looking for the
/// last `.`.  Returns `None` if the path has no extension or ends with a
/// separator.  The returned slice preserves the original casing - compare with
/// [`str::eq_ignore_ascii_case`] or call `.to_ascii_lowercase()` yourself.
///
/// # Examples
///
/// ```
/// use outil::file_extension;
///
/// assert_eq!(file_extension("https://example.com/data/tile.b3dm?v=1"), Some("b3dm"));
/// assert!(file_extension("model.GLB").map_or(false, |e| e.eq_ignore_ascii_case("glb")));
/// assert_eq!(file_extension("no_extension"), None);
/// ```
pub fn file_extension(url: &str) -> Option<&str> {
    let path = url.split('?').next().unwrap_or(url);
    let path = path.split('#').next().unwrap_or(path);
    let last_sep = path.rfind(|c| c == '/' || c == '\\').map_or(0, |i| i + 1);
    let filename = &path[last_sep..];
    let dot = filename.rfind('.')?;
    let ext = &filename[dot + 1..];
    if ext.is_empty() {
        return None;
    }
    // Return a ref into the original slice without allocating; callers that
    // need lowercase must call `.to_ascii_lowercase()` themselves.  We return
    // a sub-slice of the input so the lifetime is tied to `url`.
    //
    // NOTE: ASCII-case-fold without allocation - we return the raw slice and
    // document that extension comparisons should use `eq_ignore_ascii_case`.
    Some(ext)
}
/// Expand a URL template by replacing `{token}` placeholders with their
/// corresponding values.
///
/// Each entry in `tokens` is a `(name, value)` pair.  Any `{name}` in
/// `template` is replaced with `value`.  Unrecognised tokens are passed
/// through verbatim (e.g. `{unknown}` -> `{unknown}`).
///
/// A single scan of `template` produces the output, avoiding the intermediate
/// allocations that chained `.replace()` calls would create.
///
/// # Example
///
/// ```
/// use outil::expand_tile_url;
///
/// let url = expand_tile_url(
///     "tiles/{level}/{x}/{y}.terrain",
///     &[("level", "3"), ("x", "5"), ("y", "2")],
/// );
/// assert_eq!(url, "tiles/3/5/2.terrain");
/// ```
pub fn expand_tile_url(template: &str, tokens: &[(&str, &str)]) -> String {
    let mut out = String::with_capacity(template.len() + 16);
    let bytes = template.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'{' {
            if let Some(rel_end) = bytes[i..].iter().position(|&b| b == b'}') {
                let token = &template[i + 1..i + rel_end];
                if let Some((_, value)) = tokens.iter().find(|(name, _)| *name == token) {
                    out.push_str(value);
                } else {
                    // Unknown token: pass through verbatim including braces.
                    out.push_str(&template[i..i + rel_end + 1]);
                }
                i += rel_end + 1;
                continue;
            }
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_relative_http() {
        assert_eq!(
            resolve_url("https://example.com/tiles/model.gltf", "buffer0.bin"),
            "https://example.com/tiles/buffer0.bin"
        );
    }

    #[test]
    fn resolve_relative_file() {
        assert_eq!(
            resolve_url("/data/tiles/model.gltf", "textures/tex.png"),
            "/data/tiles/textures/tex.png"
        );
    }

    #[test]
    fn resolve_windows_path() {
        assert_eq!(
            resolve_url(
                r"C:\Users\foo\tiles/data/Models\Box\glTF\Box.gltf",
                "Box0.bin"
            ),
            r"C:\Users\foo\tiles/data/Models\Box\glTF\Box0.bin"
        );
    }

    #[test]
    fn resolve_absolute_key_passthrough() {
        assert_eq!(
            resolve_url(
                "https://example.com/tiles/model.gltf",
                "https://cdn.example.com/buf.bin"
            ),
            "https://cdn.example.com/buf.bin"
        );
    }

    #[test]
    fn resolve_root_relative() {
        assert_eq!(
            resolve_url("https://example.com/tiles/model.gltf", "/other/tex.png"),
            "/other/tex.png"
        );
    }

    #[test]
    fn resolve_strips_query_from_base() {
        assert_eq!(
            resolve_url(
                "https://example.com/tiles/model.gltf?token=abc",
                "buffer0.bin"
            ),
            "https://example.com/tiles/buffer0.bin"
        );
    }

    #[test]
    fn ext_http_with_query() {
        assert_eq!(
            file_extension("https://example.com/data/tile.b3dm?v=1"),
            Some("b3dm")
        );
    }

    #[test]
    fn ext_preserves_case() {
        assert_eq!(file_extension("model.GLB"), Some("GLB"));
    }

    #[test]
    fn ext_no_extension() {
        assert_eq!(file_extension("no_extension"), None);
    }

    #[test]
    fn ext_trailing_dot() {
        assert_eq!(file_extension("trailing."), None);
    }

    #[test]
    fn resolve_parent_dot_segment() {
        assert_eq!(
            resolve_url(
                "https://example.com/tiles/sub/manifest.json",
                "../buffers/b.bin"
            ),
            "https://example.com/tiles/buffers/b.bin"
        );
    }

    #[test]
    fn resolve_current_dot_segment() {
        assert_eq!(
            resolve_url("https://example.com/tiles/manifest.json", "./buf.bin"),
            "https://example.com/tiles/buf.bin"
        );
    }

    #[test]
    fn resolve_double_parent_dot_segment() {
        assert_eq!(
            resolve_url(
                "https://example.com/a/b/c/manifest.json",
                "../../d/file.bin"
            ),
            "https://example.com/a/d/file.bin"
        );
    }

    #[test]
    fn resolve_dot_segments_cannot_escape_root() {
        assert_eq!(
            resolve_url("https://example.com/a.json", "../../../b.bin"),
            "https://example.com/b.bin"
        );
    }

    #[test]
    fn resolve_query_embedded_scheme_is_not_absolute() {
        assert_eq!(
            resolve_url(
                "https://example.com/tiles/manifest.json",
                "sibling?next=https://other.example/x"
            ),
            "https://example.com/tiles/sibling?next=https://other.example/x"
        );
    }

    #[test]
    fn basic_substitution() {
        let result = expand_tile_url(
            "tiles/{level}/{x}/{y}.terrain",
            &[("level", "3"), ("x", "5"), ("y", "2")],
        );
        assert_eq!(result, "tiles/3/5/2.terrain");
    }

    #[test]
    fn unknown_token_passthrough() {
        let result = expand_tile_url("tiles/{level}/{unknown}", &[("level", "0")]);
        assert_eq!(result, "tiles/0/{unknown}");
    }

    #[test]
    fn empty_tokens() {
        let result = expand_tile_url("https://example.com/tiles", &[]);
        assert_eq!(result, "https://example.com/tiles");
    }

    #[test]
    fn no_tokens_in_template() {
        let result = expand_tile_url("static/url", &[("x", "1"), ("y", "2")]);
        assert_eq!(result, "static/url");
    }

    #[test]
    fn multiple_replacements() {
        let result = expand_tile_url(
            "{level}/{x}/{y}/{z}",
            &[("level", "2"), ("x", "1"), ("y", "3"), ("z", "0")],
        );
        assert_eq!(result, "2/1/3/0");
    }

    #[test]
    fn unclosed_brace_passthrough() {
        // A lone '{' with no closing '}' should be copied verbatim.
        let result = expand_tile_url("abc{def", &[("def", "X")]);
        assert_eq!(result, "abc{def");
    }
}
