/// A data attribution credit for a raster overlay data source.
///
/// Providers populate credits; the application is responsible for
/// displaying them to users as required by data provider ToS.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Credit {
    /// Display text or HTML for this credit.
    pub html: String,
    /// Whether this credit should be shown on screen (vs. info panel only).
    pub show_on_screen: bool,
}

impl Credit {
    pub fn new(html: impl Into<String>) -> Self {
        Self {
            html: html.into(),
            show_on_screen: true,
        }
    }

    pub fn info_panel_only(html: impl Into<String>) -> Self {
        Self {
            html: html.into(),
            show_on_screen: false,
        }
    }
}
