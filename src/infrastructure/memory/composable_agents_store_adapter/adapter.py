"""composable-agents Store File API adapter for per-user memory persistence.

Writes through the verified upstream endpoints::

    GET  /api/v1/store/files?prefix=<scope>     (listing — QA BUG-002)
    GET  /api/v1/store/files/memories[/<agent>]/<name>.md   (read + pre-check)
    PUT  /api/v1/store/files/memories[/<agent>]/<name>.md   (write)
    DELETE /api/v1/store/files/memories[/<agent>]/<name>.md (deletion)

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
from typing import cast

from domain.entities.memory import (
    MEMORY_ROOT,
    MemoryCredential,
    MemoryListItem,
    memory_scope_path,
    normalize_memory_name,
)
from domain.errors.memory_store import (
    MemoryIdentityError,
    MemoryNotFoundError,
    MemoryUpstreamError,
    MemoryValidationError,
)
from domain.errors.messages import ErrorMessage
from domain.logging.messages import LogMessage
from domain.ports.memory_store import MemoryStorePort
from domain.services.scope import validate_agent_slug
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

    def _store_path(self, name: str, agent_slug: str | None = None) -> str:
        """Re-validate the name and confine the outbound path to /memories/.

        Independent confinement: the caller-scope composition (REQ-012) is
        re-validated here — the stem is normalized again, the optional agent
        segment must satisfy the REQ-002 slug rule, and the final subject may
        carry AT MOST that one nesting segment before any upstream request.

        Args:
            name: Normalized memory name (stem) handed over by the use case.
            agent_slug: The caller's agent slug, or ``None`` for the flat
                personal namespace.

        Returns:
            The exact upstream path
            ``/api/v1/store/files/memories[/<agent_slug>]/<stem>.md``.

        Raises:
            MemoryValidationError: If the re-validation rejects the name or
                the composed subject.
            MemoryIdentityError: If the credential cannot be forwarded.
        """
        if agent_slug:
            validate_agent_slug(agent_slug)
        # Independent confinement: never trust the caller layer's validation.
        stem = normalize_memory_name(name)
        path = memory_scope_path(stem, agent_slug)
        subject = path[len(MEMORY_ROOT) : -len(".md")]
        if not path.startswith(MEMORY_ROOT) or subject.count("/") > 1:
            raise MemoryValidationError(ErrorMessage.MEMORY_STORE_PATH_ESCAPE)
        # Raises MemoryIdentityError before any client/request is created.
        return f"{_STORE_PATH_PREFIX}{path}"

    async def read(
        self, name: str, credential: MemoryCredential, *, agent_slug: str | None = None
    ) -> str | None:
        """Fetch the memory file content, or ``None`` on a genuine 404."""
        path = self._store_path(name, agent_slug)
        headers = _credential_headers(credential)
        async with self._open_client() as client:
            response = await self._send(client, "GET", path, headers=headers)
        if response.status_code == 404:
            return None
        # allow_list stays False here: the file-READ route answers an object.
        payload = cast("dict[str, object]", self._json_payload(response, "GET"))
        content = payload.get("content")
        if not isinstance(content, str):
            logger.warning(
                LogMessage.MEMORY_STORE_REQUEST_FAILED, "GET", "invalid_response"
            )
            raise MemoryUpstreamError(ErrorMessage.MEMORY_STORE_INVALID_RESPONSE)
        return content

    async def write(
        self,
        name: str,
        content: str,
        credential: MemoryCredential,
        *,
        agent_slug: str | None = None,
    ) -> None:
        """Persist the exact final content (never truncated)."""
        path = self._store_path(name, agent_slug)
        headers = _credential_headers(credential)
        async with self._open_client() as client:
            response = await self._send(
                client, "PUT", path, headers=headers, json={"content": content}
            )
        # allow_list stays False here: the file-WRITE route answers an object.
        payload = cast("dict[str, object]", self._json_payload(response, "PUT"))
        echoed = payload.get("content")
        if not isinstance(echoed, str):
            logger.warning(
                LogMessage.MEMORY_STORE_REQUEST_FAILED, "PUT", "invalid_response"
            )
            raise MemoryUpstreamError(ErrorMessage.MEMORY_STORE_INVALID_RESPONSE)

    async def list(
        self, credential: MemoryCredential, *, agent_slug: str | None = None
    ) -> list[MemoryListItem]:
        """List the memory files visible in the caller's scope.

        The listing goes through the composable-agents store LISTING route
        ``GET /api/v1/store/files?prefix=<scope>`` — NOT the per-file read
        route (QA BUG-002: calling ``GET .../memories[/<slug>]`` hits the
        file-content route, which answers 404 for a directory path that the
        caller previously swallowed as an "empty scope"). The real route
        answers a plain JSON array of store paths; the results are filtered
        defensively here as well so only names INSIDE the requested scope
        (flat for personal, one-segment-deep for agents) ever surface.

        Args:
            credential: The authenticated caller's own credential.
            agent_slug: The caller's agent slug, or ``None`` for the flat
                personal namespace.

        Returns:
            The listing entries (names + store paths).

        Raises:
            Same sanitization contract as :meth:`read`.
        """
        if agent_slug:
            validate_agent_slug(agent_slug)
        scope_prefix = f"{MEMORY_ROOT}{agent_slug}/" if agent_slug else MEMORY_ROOT
        headers = _credential_headers(credential)
        async with self._open_client() as client:
            response = await self._send(
                client,
                "GET",
                _STORE_PATH_PREFIX,
                headers=headers,
                params={"prefix": scope_prefix},
            )
        entries = self._listing_entries(response)
        items: list[MemoryListItem] = []
        for entry in entries:
            name = self._listing_name(entry, scope_prefix)
            if name is not None:
                items.append(MemoryListItem(name=name, path=f"{scope_prefix}{name}.md"))
        return items

    def _listing_entries(self, response: httpx.Response) -> "list[object]":
        """Decode the upstream listing payload (plain array or legacy dict).

        NOTE: the return annotation is quoted because the class namespace
        already binds the ``list`` listing METHOD above — a bare ``list[...]``
        annotation would resolve against that binding at definition time.
        """
        payload = self._json_payload(response, "GET", allow_list=True)
        if isinstance(payload, list):
            return list(payload)
        payload_dict = cast("dict[str, object]", payload)
        files = payload_dict.get("files")
        if isinstance(files, list):
            return list(files)
        logger.warning(
            LogMessage.MEMORY_STORE_REQUEST_FAILED, "GET", "invalid_response"
        )
        raise MemoryUpstreamError(ErrorMessage.MEMORY_STORE_INVALID_RESPONSE)

    @staticmethod
    def _listing_name(entry: object, scope_prefix: str) -> str | None:
        """Extract one scoped listing name or ``None`` for a foreign entry.

        Args:
            entry: One upstream listing record — the real route answers a
                plain JSON array of store paths (``str``); a legacy
                ``{"path": ..., ...}`` record is tolerated defensively.
            scope_prefix: The caller's scope prefix (``/memories/`` or
                ``/memories/<agent_slug>/``).

        Returns:
            The name relative to the scope (without ``.md``), or ``None``
                when the record does not belong to the caller's scope, is
                nested deeper than the scope namespace, or is malformed.
        """
        stored = entry.get("path") if isinstance(entry, dict) else entry
        if not isinstance(stored, str) or not stored.startswith(scope_prefix):
            return None
        name = stored[len(scope_prefix) :]
        if not name.endswith(".md") or "/" in name:
            return None  # deeper nesting never surfaces inside this scope
        stem = name[: -len(".md")]
        return stem if stem else None

    async def delete(
        self, name: str, credential: MemoryCredential, *, agent_slug: str | None = None
    ) -> None:
        """Delete one scoped memory file (absent → ``MemoryNotFoundError``).

        The upstream store's DELETE route answers ``204 No Content`` for an
        ABSENT file exactly like for a successful deletion (QA BUG-004,
        steps I2/I3) — the two cases are indistinguishable from the DELETE
        response alone, and silently reporting success for a nonexistent
        memory would break the documented not-found contract. The adapter
        therefore verifies the file EXISTS (file-read route) before the
        deletion: a 404 pre-check raises
        :class:`~domain.errors.memory_store.MemoryNotFoundError` with zero
        deletions issued; a 2xx pre-check proceeds, and a body-less 2xx
        DELETE response (QA BUG-003: the real route answers ``204 No
        Content``) is treated as success. A future upstream that answers
        ``404`` for an absent DELETE is still honoured by the kept 404
        branch, so both store contract shapes stay correct.
        """
        path = self._store_path(name, agent_slug)
        headers = _credential_headers(credential)
        async with self._open_client() as client:
            exists = await self._send(client, "GET", path, headers=headers)
            if exists.status_code == 404:
                # Absent file: surface the documented not-found error BEFORE
                # deleting (the store's DELETE would answer 204 either way).
                raise MemoryNotFoundError(
                    ErrorMessage.MEMORY_NOT_FOUND.format(name=name)
                )
            self._json_payload(exists, "GET")
            response = await self._send(client, "DELETE", path, headers=headers)
        if response.status_code == 404:
            raise MemoryNotFoundError(ErrorMessage.MEMORY_NOT_FOUND.format(name=name))
        if response.status_code == 204 or not response.content:
            return  # QA BUG-003: a body-less successful DELETE (204).
        self._json_payload(response, "DELETE")

    async def _send(
        self,
        client: httpx.AsyncClient,
        method: str,
        path: str,
        *,
        headers: dict[str, str],
        json: dict[str, str] | None = None,
        params: dict[str, str] | None = None,
    ) -> httpx.Response:
        """Perform one upstream request with sanitized failure mapping."""
        try:
            if method == "GET":
                return await client.get(path, headers=headers, params=params)
            if method == "DELETE":
                return await client.delete(path, headers=headers)
            return await client.put(path, headers=headers, json=json or {})
        except httpx.TimeoutException:
            logger.warning(LogMessage.MEMORY_STORE_REQUEST_FAILED, method, "timeout")
            raise MemoryUpstreamError(ErrorMessage.MEMORY_STORE_TIMEOUT) from None
        except httpx.HTTPError:
            logger.warning(
                LogMessage.MEMORY_STORE_REQUEST_FAILED, method, "unavailable"
            )
            raise MemoryUpstreamError(ErrorMessage.MEMORY_STORE_UNAVAILABLE) from None

    def _json_payload(
        self, response: httpx.Response, method: str, *, allow_list: bool = False
    ) -> "dict | list":
        """Validate the upstream status and decode the JSON body safely.

        Args:
            response: The upstream response.
            method: The HTTP method (failure-log category).
            allow_list: Accept a top-level JSON array (the listing route —
                QA BUG-002 — answers a plain ``list[str]`` of store paths;
                every other endpoint answers an object).
        """
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
        accepted: tuple[type, ...] = (dict, list) if allow_list else (dict,)
        if not isinstance(payload, accepted):
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
