"""CSS for the Replicant Character Lab tab (banner + step rail)."""

CSS = """
/* Make our main-webui tab stand out with a purple border echoing the logo. */
button.replicant-tabbtn {
    border: 2px solid #a64dff !important;
    border-radius: 4px !important;
    box-shadow: 0 0 6px rgba(166, 77, 255, 0.55) !important;
}

#replicant-banner { text-align: center; margin: 4px 0 14px 0; }
#replicant-banner img { width: 100%; max-width: 640px; height: auto; display: inline-block; }

#replicant-header { align-items: center; margin: 4px 0 14px 0; }
#replicant-header #replicant-banner { margin: 0; }
.replicant-tagline { font-size: 1.5em; font-weight: 700; line-height: 1.25;
    margin: 2px 0; color: #ff5fa2; }

#replicant-rail { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 12px; }
#replicant-rail button {
    flex: 1 1 0; min-width: 90px; font-size: 12px; padding: 6px 4px;
    border-radius: 8px;
}

#replicant-nav { margin-top: 14px; display: flex; justify-content: space-between; }

/* Prerequisites + its Directories/Models sub-accordion headers: 2x, bold */
.replicant-acc > .label-wrap span,
.replicant-acc > button.label-wrap span { font-size: 2em !important; font-weight: 700 !important; }

.replicant-step { border: 1px solid var(--border-color-primary);
    border-radius: 10px; padding: 16px; background: var(--background-fill-secondary); }
.replicant-step h3 { margin-top: 0; }
"""
