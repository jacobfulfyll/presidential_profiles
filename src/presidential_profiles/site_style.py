"""Shared page chrome for the dashboard and profile pages."""

from .figures import GRID, INK, INK2, SURFACE

FONT = "system-ui, -apple-system, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif"

PAGE_CSS = f"""
  :root {{
    --surface: {SURFACE}; --page: #f9f9f7; --ink: {INK}; --ink2: {INK2};
    --muted: #706d67; --grid: {GRID}; --border: rgba(11,11,11,0.14);
    --focus: #7a3d00;
  }}
  * {{ box-sizing: border-box; margin: 0; }}
  body {{ background: var(--page); color: var(--ink);
         font-family: {FONT}; line-height: 1.55; }}
  header {{ max-width: 980px; margin: 0 auto; padding: 48px 20px 8px; }}
  header h1 {{ font-size: 2rem; letter-spacing: -0.02em; }}
  header p.sub {{ color: var(--ink2); margin-top: 10px; max-width: 46rem; }}
  main {{ max-width: 980px; margin: 0 auto; padding: 8px 20px 40px; }}
  section {{ margin-top: 44px; }}
  section h2 {{ font-size: 1.28rem; letter-spacing: -0.01em; }}
  section p {{ color: var(--ink2); margin: 8px 0 14px; max-width: 46rem; }}
  .chart-scroll {{ background: var(--surface); border: 1px solid var(--border);
                   border-radius: 12px; padding: 10px 6px 6px;
                   overflow-x: auto; overflow-y: hidden; }}
  .chart {{ min-width: 640px; }}
  :where(a, button, summary, select, [tabindex]):focus-visible {{
    outline: 3px solid var(--focus); outline-offset: 3px; border-radius: 3px;
  }}
  footer {{ max-width: 980px; margin: 24px auto 60px; padding: 18px 20px 0;
            border-top: 1px solid var(--grid); color: var(--muted); font-size: 0.85rem; }}
  footer a {{ color: var(--ink2); }}
"""
