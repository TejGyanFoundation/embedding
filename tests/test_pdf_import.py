"""Endpoint-level tests. The semantic derivation itself lives in the
shrulipi-to-unicode repo (tests/test_semantics.py) — the converter is the
single source of truth for the pdf-import.v2 semantics."""
import pytest
from fastapi import HTTPException

from app.pdf_import import _check_authorization
from app.pdf_import import enrich_document


def test_enrich_document_is_the_converter_implementation():
    from shrulipi_to_unicode import semantics

    assert enrich_document is semantics.enrich_document


def test_authorization_open_when_no_key(monkeypatch):
    monkeypatch.delenv("PDF_IMPORT_API_KEY", raising=False)
    _check_authorization(None)  # must not raise


def test_authorization_enforced_when_key_set(monkeypatch):
    monkeypatch.setenv("PDF_IMPORT_API_KEY", "sekrit")
    _check_authorization("Bearer sekrit")  # must not raise
    with pytest.raises(HTTPException):
        _check_authorization("Bearer wrong")
    with pytest.raises(HTTPException):
        _check_authorization(None)
