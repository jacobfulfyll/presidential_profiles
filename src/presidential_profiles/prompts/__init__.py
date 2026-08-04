"""Frozen prompt artifacts for the paid LLM annotation passes.

Rubrics, JSON Schemas, and per-speech context builders live here so the
request-construction machinery in ``annotate.py`` stays about batching, cost,
and provenance — never about prompt wording. Each module pins a
``PROMPT_VERSION`` and is meant to be frozen once a paid run has been paid for;
``FieldSpec.prompt_hash()`` fingerprints the exact bytes that were sent.
"""
