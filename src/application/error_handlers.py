"""FastAPI exception handlers for all domain errors.

Each handler reads the exception's own ``status_code`` and ``detail``,
logs at the appropriate level, and returns a uniform JSON response.
The ``exc`` parameters are deliberately typed as :class:`Exception` to
satisfy Starlette's ``ExceptionHandler`` callable contract; the registry in
``main.py`` only ever binds them to :class:`DomainError` subclasses, which
``_domain_response`` guards at the boundary.
"""

import logging

from fastapi import Request
from fastapi.responses import JSONResponse

from domain.errors.base import DomainError
from domain.logging.messages import LogMessage

logger = logging.getLogger(__name__)


def _domain_response(
    exc: Exception, message: LogMessage, *, level: int
) -> JSONResponse:
    """Log a domain error at ``level`` and render the uniform JSON response.

    Args:
        exc: The exception dispatched by the exception-handler registry. It
            is always a :class:`DomainError` subclass (see ``main.py``); the
            broad parameter type exists solely for FastAPI's callback contract.
        message: Centralized log template for this error category.
        level: Standard logging level for the category (``logging.WARNING`` /
            ``logging.ERROR``).

    Returns:
        The uniform ``{"detail": ...}`` JSON response with the error's own
        status code.

    Raises:
        TypeError: If a non-domain error reaches a registered handler
            (unreachable by construction; guards against accidental
            misregistration).
    """
    if not isinstance(exc, DomainError):
        raise TypeError(f"registered handler received a non-domain error: {exc!r}")
    logger.log(level, message, exc.detail)
    return JSONResponse(
        status_code=int(exc.status_code), content={"detail": exc.detail}
    )


def storage_not_found_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Render a missing-storage-object error as 404."""
    return _domain_response(exc, LogMessage.LOG_STORAGE_ERROR, level=logging.WARNING)


def storage_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Render a storage failure as 503."""
    return _domain_response(exc, LogMessage.LOG_STORAGE_ERROR, level=logging.ERROR)


def unsupported_format_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Render an unsupported document format as 415."""
    return _domain_response(exc, LogMessage.LOG_DOCUMENT_ERROR, level=logging.WARNING)


def document_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Render a document read failure as 400."""
    return _domain_response(exc, LogMessage.LOG_DOCUMENT_ERROR, level=logging.ERROR)


def vector_bad_request_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Render a vector-store configuration error as 400."""
    return _domain_response(
        exc, LogMessage.LOG_VECTOR_STORE_ERROR, level=logging.WARNING
    )


def vector_store_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Render a vector-store failure as 503."""
    return _domain_response(exc, LogMessage.LOG_VECTOR_STORE_ERROR, level=logging.ERROR)


def classical_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Render a classical RAG configuration error as 400."""
    return _domain_response(exc, LogMessage.LOG_CLASSICAL_ERROR, level=logging.WARNING)


def indexing_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Render an indexing failure as 503."""
    return _domain_response(exc, LogMessage.LOG_INDEXING_ERROR, level=logging.ERROR)


def file_validation_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Render a file validation error as 400."""
    return _domain_response(exc, LogMessage.LOG_FILE_ERROR, level=logging.WARNING)


def file_too_large_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Render an oversized upload as 413."""
    return _domain_response(exc, LogMessage.LOG_FILE_ERROR, level=logging.WARNING)


def file_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Render a generic file failure as 400."""
    return _domain_response(exc, LogMessage.LOG_FILE_ERROR, level=logging.WARNING)


def dependency_not_initialized_handler(
    _request: Request, exc: Exception
) -> JSONResponse:
    """Render an uninitialized dependency as 503."""
    return _domain_response(exc, LogMessage.LOG_CONFIG_ERROR, level=logging.ERROR)


def config_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Render a configuration error as 400."""
    return _domain_response(exc, LogMessage.LOG_CONFIG_ERROR, level=logging.WARNING)


def domain_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Render any unregistered domain error as its own status code."""
    return _domain_response(
        exc, LogMessage.LOG_UNHANDLED_DOMAIN_ERROR, level=logging.ERROR
    )


def invalid_api_key_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Render an invalid API key as 401."""
    return _domain_response(exc, LogMessage.LOG_INVALID_API_KEY, level=logging.WARNING)


def authentication_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Map :class:`AuthenticationError` (raised by verify_credentials) to 401."""
    return _domain_response(exc, LogMessage.LOG_INVALID_API_KEY, level=logging.WARNING)


def llm_not_configured_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Map :class:`LlmNotConfiguredError` to 422 (user has no LLM provider)."""
    return _domain_response(exc, LogMessage.LOG_CONFIG_ERROR, level=logging.WARNING)


def mcp_invalid_stored_config_handler(
    _request: Request, exc: Exception
) -> JSONResponse:
    """Map a corrupted stored MCP config (invalid ``auth_mode``) to 500.

    A structured server-side error with its OWN centralized log message — the
    generic "Unhandled domain error" line would say nothing about the actual
    corruption class.
    """
    return _domain_response(
        exc, LogMessage.LOG_MCP_INVALID_STORED_CONFIG, level=logging.ERROR
    )
