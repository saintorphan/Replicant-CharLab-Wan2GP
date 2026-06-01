"""CSS for the Replicant Character Lab tab (banner + step rail)."""

CSS = """
#replicant-banner { text-align: center; margin: 4px 0 14px 0; }
#replicant-banner img { width: 100%; max-width: 1580px; height: auto; display: inline-block; }

#replicant-rail { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 12px; }
#replicant-rail button {
    flex: 1 1 0; min-width: 90px; font-size: 12px; padding: 6px 4px;
    border-radius: 8px;
}

#replicant-nav { margin-top: 14px; display: flex; justify-content: space-between; }

.replicant-step { border: 1px solid var(--border-color-primary);
    border-radius: 10px; padding: 16px; background: var(--background-fill-secondary); }
.replicant-step h3 { margin-top: 0; }
"""
