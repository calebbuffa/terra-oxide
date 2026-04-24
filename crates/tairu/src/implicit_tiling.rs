pub mod utlx {
    use outil::{expand_tile_url, resolve_url};
    use zukei::{OctreeTileID, QuadtreeTileID, S2CellId};

    /// Resolve a 3D Tiles implicit tiling URL template for a quadtree tile.
    ///
    /// Replaces `{level}`, `{x}`, and `{y}` in `url_template`, then joins the
    /// result against `base_url` (relative paths are resolved relative to
    /// `base_url`'s directory).
    pub fn resolve_url_quad(base_url: &str, url_template: &str, tile: QuadtreeTileID) -> String {
        let level = tile.level.to_string();
        let x = tile.x.to_string();
        let y = tile.y.to_string();
        let expanded = expand_tile_url(url_template, &[("level", &level), ("x", &x), ("y", &y)]);
        resolve_url(base_url, &expanded)
    }

    /// Resolve a 3D Tiles implicit tiling URL template for an octree tile.
    ///
    /// Replaces `{level}`, `{x}`, `{y}`, and `{z}`.
    pub fn resolve_url_oct(base_url: &str, url_template: &str, tile: OctreeTileID) -> String {
        let level = tile.level.to_string();
        let x = tile.x.to_string();
        let y = tile.y.to_string();
        let z = tile.z.to_string();
        let expanded = expand_tile_url(
            url_template,
            &[("level", &level), ("x", &x), ("y", &y), ("z", &z)],
        );
        resolve_url(base_url, &expanded)
    }

    /// Resolve a 3D Tiles implicit tiling URL template for an S2 cell.
    ///
    /// Replaces `{s2cellid}` with the S2 token string (compact hex, trailing
    /// zeros stripped), then joins the result against `base_url`.
    pub fn resolve_url_s2(base_url: &str, url_template: &str, cell: S2CellId) -> String {
        let token = cell.to_token();
        let expanded = expand_tile_url(url_template, &[("s2cellid", &token)]);
        resolve_url(base_url, &expanded)
    }
}

#[cfg(test)]
mod tests {

    use super::utlx::*;
    use zukei::{OctreeTileID, QuadtreeTileID, S2CellId};

    #[test]
    fn test_resolve_url_quad() {
        let url = resolve_url_quad(
            "https://example.com/tileset/tileset.json",
            "subtrees/{level}/{x}/{y}.subtree",
            QuadtreeTileID::new(3, 5, 2),
        );
        assert_eq!(url, "https://example.com/tileset/subtrees/3/5/2.subtree");
    }

    #[test]
    fn test_resolve_url_oct() {
        let url = resolve_url_oct(
            "https://example.com/tileset.json",
            "subtrees/{level}/{x}/{y}/{z}.subtree",
            OctreeTileID::new(2, 1, 3, 0),
        );
        assert_eq!(url, "https://example.com/subtrees/2/1/3/0.subtree");
    }

    #[test]
    fn test_resolve_url_absolute_passthrough() {
        let url = resolve_url_quad(
            "https://example.com/tileset.json",
            "https://cdn.example.com/subtrees/{level}/{x}/{y}.subtree",
            QuadtreeTileID::new(0, 0, 0),
        );
        assert!(url.starts_with("https://cdn.example.com/"));
    }

    #[test]
    fn test_resolve_url_s2() {
        // Face 0, level 0 token is "1".
        let cell = S2CellId::from_raw(0x1000000000000000);
        let url = resolve_url_s2(
            "https://example.com/tileset.json",
            "subtrees/{s2cellid}.subtree",
            cell,
        );
        assert_eq!(url, "https://example.com/subtrees/1.subtree");
    }
}
