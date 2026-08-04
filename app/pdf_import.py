"""Async structured PDF import.

Implements the contract Django's `_submit_to_pdf_service` expects:

    POST /process-document-url
      {"url": ..., "document_id": ..., "document_type": ..., "callback_url": ...}
      -> {"identifier": ..., "status": "accepted"}   (immediately)

A background worker then downloads the PDF, converts it with
`shrulipi-to-unicode` into a ``pdf-import.v2`` payload, enriches it with the
converter's own semantic derivation (sessions/chapters/sections, speaker
turns, bhajan/meditation typing, contributors), and POSTs the result to
``callback_url``. Failures are reported to the same callback with
``status: "failed"`` so the document does not stay stuck in progress.
"""
import logging
import os
import tempfile
import time
import uuid
from typing import Optional

import requests
from fastapi import APIRouter, BackgroundTasks, Header, HTTPException
from pydantic import BaseModel

from shrulipi_to_unicode.jsonexport import build_document
from shrulipi_to_unicode.semantics import enrich_document

logger = logging.getLogger(__name__)

router = APIRouter()

DOWNLOAD_TIMEOUT_SECONDS = int(os.getenv("PDF_DOWNLOAD_TIMEOUT", "120"))
CALLBACK_TIMEOUT_SECONDS = int(os.getenv("PDF_CALLBACK_TIMEOUT", "120"))
CALLBACK_ATTEMPTS = int(os.getenv("PDF_CALLBACK_ATTEMPTS", "4"))
CALLBACK_RETRY_DELAY_SECONDS = float(os.getenv("PDF_CALLBACK_RETRY_DELAY", "8"))
MAX_PDF_BYTES = int(os.getenv("PDF_MAX_BYTES", str(200 * 1024 * 1024)))

class ProcessDocumentRequest(BaseModel):
    url: str
    document_id: str
    callback_url: str
    document_type: Optional[str] = None
    # Per-job secret issued by the submitting backend; echoed back on the
    # callback as X-Callback-Token so the callback endpoint can verify the
    # result really comes from this job.
    callback_token: Optional[str] = None


class ProcessDocumentResponse(BaseModel):
    identifier: str
    status: str = "accepted"


def _check_authorization(authorization: Optional[str]) -> None:
    """Reject the request when an API key is configured and does not match."""
    expected_key = os.getenv("PDF_IMPORT_API_KEY")
    if not expected_key:
        return
    if authorization != f"Bearer {expected_key}":
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@router.post("/process-document-url", response_model=ProcessDocumentResponse)
def process_document_url(
    request: ProcessDocumentRequest,
    background_tasks: BackgroundTasks,
    authorization: Optional[str] = Header(default=None),
) -> ProcessDocumentResponse:
    _check_authorization(authorization)
    identifier = f"shrulipi-job-{uuid.uuid4().hex[:16]}"
    background_tasks.add_task(
        process_and_callback,
        request.url,
        request.document_id,
        request.callback_url,
        identifier,
        request.callback_token,
    )
    logger.info(
        "Accepted PDF import job identifier=%s document_id=%s",
        identifier,
        request.document_id,
    )
    return ProcessDocumentResponse(identifier=identifier)


def process_and_callback(
    pdf_url: str,
    document_id: str,
    callback_url: str,
    identifier: str,
    callback_token: Optional[str] = None,
) -> None:
    """Worker: download -> convert -> derive hierarchy -> POST callback."""
    try:
        payload = convert_pdf_url(pdf_url, identifier)
    except Exception as exc:  # noqa: BLE001 - any failure must reach the callback
        logger.exception(
            "PDF import failed identifier=%s document_id=%s", identifier, document_id
        )
        payload = {
            "schema_version": "pdf-import.v2",
            "status": "failed",
            "identifier": identifier,
            "job_id": identifier,
            "error": {"message": str(exc)},
        }

    # The token authenticates this job to the callback endpoint; sent as a
    # header (not in the URL) so it stays out of access logs.
    headers = {}
    if callback_token:
        headers["X-Callback-Token"] = callback_token

    # The receiving backend may hit transient contention (e.g. deadlocks when
    # several imports land at once) — retry 5xx and connection errors.
    for attempt in range(1, CALLBACK_ATTEMPTS + 1):
        try:
            response = requests.post(
                callback_url,
                json=payload,
                timeout=CALLBACK_TIMEOUT_SECONDS,
                headers=headers,
            )
            response.raise_for_status()
            logger.info(
                "Delivered PDF import callback identifier=%s document_id=%s "
                "status=%s",
                identifier,
                document_id,
                payload.get("status"),
            )
            return
        except requests.RequestException as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            retryable = status is None or status >= 500
            if not retryable or attempt == CALLBACK_ATTEMPTS:
                logger.exception(
                    "PDF import callback delivery failed identifier=%s "
                    "callback_url=%s",
                    identifier,
                    callback_url,
                )
                return
            logger.warning(
                "Callback delivery attempt %d/%d failed (%s), retrying in %.0fs",
                attempt,
                CALLBACK_ATTEMPTS,
                exc,
                CALLBACK_RETRY_DELAY_SECONDS,
            )
            time.sleep(CALLBACK_RETRY_DELAY_SECONDS)


def convert_pdf_url(pdf_url: str, identifier: str) -> dict:
    """Download *pdf_url* and return the enriched ``pdf-import.v2`` payload."""
    with tempfile.NamedTemporaryFile(suffix=".pdf") as handle:
        _download_pdf(pdf_url, handle)
        payload = build_document(handle.name)

    payload["identifier"] = identifier
    payload["job_id"] = identifier
    enrich_document(payload)
    return payload


DOWNLOAD_ATTEMPTS = int(os.getenv("PDF_DOWNLOAD_ATTEMPTS", "5"))
DOWNLOAD_RETRY_DELAY_SECONDS = float(os.getenv("PDF_DOWNLOAD_RETRY_DELAY", "5"))


def _download_pdf(pdf_url: str, handle) -> None:
    # The submitting backend fires the job right after saving the file, so
    # the CDN/S3 object may not be servable yet — retry 404/403/5xx briefly.
    last_error: Optional[Exception] = None
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            _download_pdf_once(pdf_url, handle)
            return
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            status = getattr(getattr(exc, "response", None), "status_code", None)
            retryable = status in (403, 404) or status is None or status >= 500
            if not retryable or attempt == DOWNLOAD_ATTEMPTS:
                raise
            logger.warning(
                "PDF download attempt %d/%d failed (%s), retrying in %.0fs",
                attempt,
                DOWNLOAD_ATTEMPTS,
                exc,
                DOWNLOAD_RETRY_DELAY_SECONDS,
            )
            handle.seek(0)
            handle.truncate()
            time.sleep(DOWNLOAD_RETRY_DELAY_SECONDS)
    if last_error is not None:
        raise last_error


def _download_pdf_once(pdf_url: str, handle) -> None:
    response = requests.get(
        pdf_url, stream=True, timeout=DOWNLOAD_TIMEOUT_SECONDS
    )
    response.raise_for_status()
    total = 0
    for chunk in response.iter_content(chunk_size=1024 * 1024):
        total += len(chunk)
        if total > MAX_PDF_BYTES:
            raise ValueError(f"PDF exceeds the {MAX_PDF_BYTES} byte limit")
        handle.write(chunk)
    handle.flush()
    if total == 0:
        raise ValueError("Downloaded PDF is empty")
