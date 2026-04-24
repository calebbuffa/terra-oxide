// terrain-server serves a Cesium terrain tileset produced by `banin`.
//
// Usage:
//
//	terrain-server [--dir <tiles-dir>] [--port <port>] [--host <host>]
//
// The tile directory must contain a layer.json and tiles laid out as:
//
//	<dir>/<zoom>/<x>/<y>.terrain   (gzip-compressed)
//
// CesiumJS expects:
//   - layer.json:        Content-Type: application/json
//   - *.terrain tiles:   Content-Type: application/octet-stream
//     Content-Encoding: gzip   (files are stored pre-compressed)
//   - All responses:     Access-Control-Allow-Origin: *
//
// A browser viewer is available at http://localhost:<port>/viewer
package main

import (
	_ "embed"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

//go:embed viewer.html
var viewerHTML []byte

func main() {
	dir := flag.String("dir", "tiles", "directory containing layer.json and terrain tiles")
	port := flag.Int("port", 8080, "TCP port to listen on")
	host := flag.String("host", "127.0.0.1", "host address to bind (default localhost; use 0.0.0.0 for LAN access)")
	flag.Parse()

	abs, err := filepath.Abs(*dir)
	if err != nil {
		log.Fatalf("cannot resolve tile directory: %v", err)
	}
	if _, err := os.Stat(filepath.Join(abs, "layer.json")); os.IsNotExist(err) {
		log.Fatalf("no layer.json found in %s — has banin been run yet?", abs)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/viewer", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Write(viewerHTML)
	})
	mux.Handle("/", newTerrainHandler(abs))

	addr := fmt.Sprintf("%s:%d", *host, *port)
	srv := &http.Server{
		Addr:         addr,
		Handler:      mux,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 30 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	log.Printf("terrain-server listening on http://%s", addr)
	log.Printf("tile directory: %s", abs)
	log.Printf("CesiumTerrainProvider URL: http://localhost:%d", *port)
	log.Printf("Viewer: http://localhost:%d/viewer", *port)
	if err := srv.ListenAndServe(); err != nil {
		log.Fatal(err)
	}
}

// terrainHandler serves files from the tile directory with appropriate
// content-type and CORS headers.
type terrainHandler struct {
	dir string
}

func newTerrainHandler(dir string) *terrainHandler {
	return &terrainHandler{dir: dir}
}

func (h *terrainHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	// Always allow cross-origin access — CesiumJS loads tiles from JS.
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Accept-Encoding, Content-Type")

	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusNoContent)
		return
	}
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Clean the URL path and map it to a file.
	urlPath := filepath.Clean(strings.TrimPrefix(r.URL.Path, "/"))
	if urlPath == "." {
		urlPath = "layer.json"
	}

	// Block path traversal attempts.
	absPath := filepath.Join(h.dir, urlPath)
	if !strings.HasPrefix(absPath, h.dir+string(filepath.Separator)) && absPath != h.dir {
		http.Error(w, "forbidden", http.StatusForbidden)
		return
	}

	info, err := os.Stat(absPath)
	if os.IsNotExist(err) || (err == nil && info.IsDir()) {
		http.NotFound(w, r)
		return
	}
	if err != nil {
		http.Error(w, "internal server error", http.StatusInternalServerError)
		return
	}

	switch {
	case urlPath == "layer.json":
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Cache-Control", "no-cache")

	case strings.HasSuffix(urlPath, ".terrain"):
		// Tiles are stored pre-gzip-compressed by banin.
		// Setting Content-Encoding: gzip tells the client the body is gzip
		// so it decompresses transparently — do NOT double-compress.
		w.Header().Set("Content-Type", "application/octet-stream")
		w.Header().Set("Content-Encoding", "gzip")
		// Force revalidation on every request so stale cached tiles are never
		// served.  no-cache + no-store ensures the browser always fetches fresh.
		w.Header().Set("Cache-Control", "no-cache, no-store, must-revalidate")
		w.Header().Set("Pragma", "no-cache")
		w.Header().Set("Expires", "0")

	default:
		// Anything else (e.g. a debug index page) gets plain octet-stream.
		w.Header().Set("Content-Type", "application/octet-stream")
	}

	f, err := os.Open(absPath)
	if err != nil {
		http.Error(w, "cannot open file", http.StatusInternalServerError)
		return
	}
	defer f.Close()

	log.Printf("%s %s", r.Method, r.URL.Path)
	http.ServeContent(w, r, urlPath, info.ModTime(), f)
}
