"""CSS for the Replicant Character Lab tab (banner + step rail)."""

CSS = """
/* Make our main-webui tab stand out with a purple border echoing the logo. */
button.replicant-tabbtn {
    border: 2px solid #a64dff !important;
    border-radius: 4px !important;
    box-shadow: 0 0 6px rgba(166, 77, 255, 0.55) !important;
}

#replicant-banner { text-align: center; margin: 4px 0 14px 0; }
#replicant-banner img { width: 56%; max-width: 890px; height: auto; display: inline-block; }

.replicant-hidden { display: none !important; }
/* Pull the lab content flush to the tab bar — no gap above the logo. */
#replicant-root { padding-top: 0 !important; margin-top: 0 !important; gap: 0; }
#replicant-header { align-items: center; margin: 0 0 14px 0; }
/* Clear Wizard column: centered, narrower button, nudged up; GitHub link below. */
#replicant-clearcol { align-items: center; transform: translateY(-8px); }
#replicant-clearbtn { width: 100%; background: #d32f2f !important; color: #fff !important;
    border-color: #d32f2f !important; }
#replicant-clearbtn:hover { background: #b71c1c !important; border-color: #b71c1c !important; }
.replicant-ghlink { text-align: center; margin: 0; padding: 0; font-size: 1.5em;
    line-height: 1; }
/* Kill the gradio block padding around the link and the column's flex gap. */
.replicant-ghwrap { padding: 0 !important; margin: 0 !important; min-height: 0 !important;
    border: none !important; }
#replicant-bannercol { gap: 0 !important; }
#replicant-bannercol #replicant-banner { margin-bottom: 0 !important; }
.replicant-ghlink a { color: #e83e8c; text-decoration: none; }
.replicant-ghlink a:hover { text-decoration: underline; }
#replicant-header #replicant-banner { margin: 0; text-align: center; }
/* Stretched to fill the column width at a fixed height (intentionally wider than
   the native 4.03:1 ratio — wider, not taller). */
#replicant-header #replicant-banner img { width: 100%; max-width: 100%; height: 175px;
    object-fit: fill; margin: 0 auto; display: block; }
.replicant-taglines { display: flex; flex-direction: column; justify-content: center;
    align-items: center; gap: 44px; height: 100%; text-align: center; }
.replicant-tagline { font-size: 1.3em; font-weight: 700; font-style: italic;
    line-height: 1.25; color: #e83e8c; }

#replicant-rail { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 12px; }
#replicant-rail button {
    flex: 1 1 0; min-width: 90px; font-size: 24px; padding: 6px 4px;
    border-radius: 8px;
}

#replicant-nav { margin-top: 14px; display: flex; justify-content: space-between; }

/* OrphanSuite + its folders/models sub-accordion headers: 2x, bold */
.replicant-acc > .label-wrap span,
.replicant-acc > button.label-wrap span { font-size: 2em !important; font-weight: 700 !important; }

/* Touch Up: options column is a fixed-height scroll box (Run + actions are in their
   own row below it); results strip scrolls sideways. */
#replicant-inpaint-opts { height: 720px; max-height: 720px;
    overflow-y: auto; overflow-x: hidden; }
#replicant-inpaint-out .grid-wrap { overflow-x: auto !important; }
#replicant-inpaint-out .grid-container {
    display: flex !important; flex-wrap: nowrap !important; }
#replicant-inpaint-out .thumbnail-item { flex: 0 0 auto !important; height: 600px !important;
    width: auto !important; }

.replicant-step { border: 1px solid var(--border-color-primary);
    border-radius: 10px; padding: 16px; background: var(--background-fill-secondary); }
.replicant-step h3 { margin-top: 0; }

/* Per-pose "Set as Base" (➕) — CSS hover tooltip. */
.replicant-pose-setbase { position: relative; overflow: visible !important; }
.replicant-pose-setbase:hover::after {
    content: "Set as Base"; position: absolute; bottom: calc(100% + 4px); left: 50%;
    transform: translateX(-50%); background: #222; color: #fff; padding: 2px 7px;
    border-radius: 4px; font-size: 11px; line-height: 1.4; white-space: nowrap;
    z-index: 60; pointer-events: none; box-shadow: 0 1px 4px rgba(0,0,0,0.4); }
"""
