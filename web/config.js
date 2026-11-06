// The one backend setting. Empty means the same origin that served this page, which is
// the right value when the page comes from the backend itself, including through a tunnel.
// If the page is hosted elsewhere, put the backend's full URL here and start the backend
// with KSHANA_CORS_ORIGIN set to the page's origin.
window.KSHANA_API = "";
