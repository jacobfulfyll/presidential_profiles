"""Context-specific escaping helpers for generated HTML."""

import json


def json_for_script(value) -> str:
    """Serialize JSON for a literal ``<script>`` block.

    ``json.dumps`` leaves ``</script>`` intact, so data containing that sequence
    can terminate the surrounding element before JavaScript parses the JSON.
    Escape the HTML-significant characters while preserving the values the
    browser receives after JavaScript string decoding.
    """
    return (
        json.dumps(value)
        .replace("&", r"\u0026")
        .replace("<", r"\u003c")
        .replace(">", r"\u003e")
        .replace("\u2028", r"\u2028")
        .replace("\u2029", r"\u2029")
    )
