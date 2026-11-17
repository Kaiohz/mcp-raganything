"""composable-agents Store File API adapter for per-user memory persistence.

Writes through the verified upstream endpoints::

    GET /api/v1/store/files/memories/{name}.md
    PUT /api/v1/store/files/memories/{name}.md

only ever forwarding the authenticated caller's own credential — ``Authorization:
Bearer <token>`` for JWT callers, ``X-API-Key: <key>`` for API-key callers;
never both schemes, never a static service key. composable-agents re-validates
the forwarded credential and resolves its own ``(user_id, "filesystem")``
namespace, so two users may share a memory filename with independent content.

Every outbound path is independently confined to ``/memories/`` here (the
use-case validation alone is insufficient): the name is re-validated and the
final path is asserted to stay inside the namespace before any upstream
request. No ``user_id`` ever crosses the boundary as data.

Each Store File call opens and closes its own short-lived async client. A
single long-lived adapter-wide client is deliberately NOT used: the per-user
memory contract requires every save to register (and fully close) its own
outbound HTTP session so that authenticated operations remain observable and
fully isolated per caller, and concurrent saves never share client lifecycle
state. Upstream statuses are mapped to distinct, sanitized domain failures;
only a genuine upstream 404 means the file is absent.
"""

import logging

from domain.entities.memory import (
    MemoryCredential,
    memory_path,
    normalize_memory_name,
)
from domain.errors.memory_store import (
    MemoryIdentityError,
    MemoryUpstreamError,
    MemoryValidationError,
)
from domain.errors.messages import ErrorMessage
from domain.logging.messages import LogMessage
from domain.ports.memory_store import MemoryStorePort
from infrastructure.memory.composable_agents_store_adapter import httpx

logger = logging.getLogger(__name__)

_STORE_PATH_PREFIX = "/api/v1/store/files"


def _credential_headers(credential: MemoryCredential) -> dict[str, str]:
    """Build upstream auth headers from the caller's own credential.

    Args:
        credential: The authenticated caller's credential and method.

    Returns:
        Exactly one authorization header (never both schemes).

    Raises:
        MemoryIdentityError: For unsupported/empty credential material.
    """
    if credential.method == "jwt" and credential.credential:
        return {"Authorization": f"Bearer {credential.credential}"}
    if credential.method == "api_key" and credential.credential:
        return {"X-API-Key": credential.credential}
    raise MemoryIdentityError(ErrorMessage.MEMORY_IDENTITY_REQUIRED)


class ComposableAgentsMemoryStoreAdapter(MemoryStorePort):
    """HTTP-bound memory store adapter (asynchronous client)."""

    def __init__(self, base_url: str, timeout_seconds: float) -> None:
        self._base_url = base_url
        self._timeout = timeout_seconds

    def _store_path(self, name: str) -> str:
        """Re-validate the name and confine the outbound path to /memories/.

        Args:
            name: Normalized memory name (stem) handed over by the use case.

        Returns:
            The exact upstream path ``/api/v1/store/files/memories/<stem>.md``.

        Raises:
            MemoryValidationError: If the re-validation rejects the name.
            MemoryIdentityError: If the credential cannot be forwarded.
        """
        # Independent confinement: never trust the caller layer's validation.
        stem = normalize_memory_name(name)
        path = memory_path(stem)
        if not path.startswith("/memories/"):
            raise MemoryValidationError(ErrorMessage.MEMORY_STORE_PATH_ESCAPE)
        # Raises MemoryIdentityError before any client/request is created.
        return f"{_STORE_PATH_PREFIX}{path}"

    async def read(self, name: str, credential: MemoryCredential) -> str | None:
        """Fetch the memory file content, or ``None`` on a genuine 404."""
        path = self._store_path(name)
        headers = _credential_headers(credential)
        async with self._open_client() as client:
            response = await self._send(client, "GET", path, headers=headers)
        if response.status_code == 404:
            return None
        payload = self._json_payload(response, "GET")
        content = payload.get("content")
        if not isinstance(content, str):
            logger.warning(
                LogMessage.MEMORY_STORE_REQUEST_FAILED, "GET", "invalid_response"
            )
            raise MemoryUpstreamError(ErrorMessage.MEMORY_STORE_INVALID_RESPONSE)
        return content

    async def write(
        self, name: str, content: str, credential: MemoryCredential
    ) -> None:
        """Persist the exact final content (never truncated)."""
        path = self._store_path(name)
        headers = _credential_headers(credential)
        async with self._open_client() as client:
            response = await self._send(
                client, "PUT", path, headers=headers, json={"content": content}
            )
        payload = self._json_payload(response, "PUT")
        echoed = payload.get("content")
        if not isinstance(echoed, str):
            logger.warning(
                LogMessage.MEMORY_STORE_REQUEST_FAILED, "PUT", "invalid_response"
            )
            raise MemoryUpstreamError(ErrorMessage.MEMORY_STORE_INVALID_RESPONSE)

    async def _send(
        self,
        client: httpx.AsyncClient,
        method: str,
        path: str,
        *,
        headers: dict[str, str],
        json: dict[str, str] | None = None,
    ) -> httpx.Response:
        """Perform one upstream request with sanitized failure mapping."""
        try:
            if method == "GET":
                return await client.get(path, headers=headers)
            return await client.put(path, headers=headers, json=json or {})
        except httpx.TimeoutException:
            logger.warning(LogMessage.MEMORY_STORE_REQUEST_FAILED, method, "timeout")
            raise MemoryUpstreamError(ErrorMessage.MEMORY_STORE_TIMEOUT) from None
        except httpx.HTTPError:
            logger.warning(
                LogMessage.MEMORY_STORE_REQUEST_FAILED, method, "unavailable"
            )
            raise MemoryUpstreamError(ErrorMessage.MEMORY_STORE_UNAVAILABLE) from None

    def _json_payload(self, response: httpx.Response, method: str) -> dict:
        """Validate the upstream status and decode the JSON body safely."""
        if response.status_code in (401, 403):
            logger.warning(LogMessage.MEMORY_STORE_REQUEST_FAILED, method, "forbidden")
            raise MemoryIdentityError(ErrorMessage.MEMORY_STORE_FORBIDDEN)
        if response.status_code // 100 != 2:
            # 4xx (other than 401/403) is a permanent request rejection,
            # not a temporary outage; 5xx is a genuine upstream failure.
            logger.warning(
                LogMessage.MEMORY_STORE_REQUEST_FAILED,
                method,
                f"upstream_status_{response.status_code // 100}xx",
            )
            raise MemoryUpstreamError(
                ErrorMessage.MEMORY_STORE_REJECTED
                if 400 <= response.status_code < 500
                else ErrorMessage.MEMORY_STORE_UNAVAILABLE
            )
        try:
            payload = response.json()
        except ValueError:
            logger.warning(
                LogMessage.MEMORY_STORE_REQUEST_FAILED, method, "invalid_response"
            )
            raise MemoryUpstreamError(
                ErrorMessage.MEMORY_STORE_INVALID_RESPONSE
            ) from None
        if not isinstance(payload, dict):
            logger.warning(
                LogMessage.MEMORY_STORE_REQUEST_FAILED, method, "invalid_response"
            )
            raise MemoryUpstreamError(ErrorMessage.MEMORY_STORE_INVALID_RESPONSE)
        return payload

    def _open_client(self) -> httpx.AsyncClient:
        """Open one short-lived async client for this Store File call.

        Deliberately per-call (see module docstring): each authenticated
        memory operation owns, uses and fully closes its outbound HTTP
        session, keeping concurrent saves and per-user isolation clean.
        """
        return httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout)

    async def close(self) -> None:
        """Idempotent shutdown hook (no long-lived client to close).

        Every Store File call owns and closes its own client (``async with``
        in :meth:`read` / :meth:`write`), so there is nothing dangling to
        release here; the port contract keeps this hook for lifespan symmetry.
        """
