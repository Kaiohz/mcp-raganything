"""auth_mode credential-injection rule for the MCP registry (REQ-004).

Pure domain rule: given the configured ``auth_mode`` of an MCP server (an
incoming validation request or a stored registry entry) and the authenticated
:class:`~domain.entities.auth.auth_context.AuthContext` of the originating
request, return the credential headers the backend must inject when
loading/validating that server's tools. The credential source travels as a
plain domain entity — no contextvar, transport or framework dependency; the
connection-dict assembly (placeholder resolution, merge order) stays at the
transport boundary (the validate route / the consuming service's loader).
"""

from domain.entities.auth.auth_context import AuthContext
from domain.entities.mcp_server_config import McpAuthMode


def user_credential_headers(
    auth_mode: McpAuthMode | None,
    context: AuthContext | None,
) -> dict[str, str]:
    """Return the credential headers to inject for ``auth_mode: "user_token"``.

    The connected user's credential is transmitted as:

    * ``Authorization: Bearer <jwt>`` when the originating request was
      JWT-authenticated (OAuth/JWT);
    * ``X-API-Key: <key>`` when it was API-key-authenticated;
    * nothing when the server uses the default ``headers`` mode or when no
      user credential exists (unattended runs / legacy master-key path) — the
      caller's empty-value dropping then never transmits a credential
      (``resolve_headers_drop_empty`` semantics, REQ-004 error case).

    Args:
        auth_mode: The server's ``auth_mode`` (``HEADERS`` default or
            ``USER_TOKEN``).
        context: The authenticated context of the originating request (``None``
            when the request carries no user credential).

    Returns:
        The single injected credential header (empty mapping = no injection).
    """
    if (
        auth_mode != McpAuthMode.USER_TOKEN
        or context is None
        or not context.raw_credential
    ):
        return {}
    if context.method == "jwt":
        return {"Authorization": f"Bearer {context.raw_credential}"}
    if context.method == "api_key":
        return {"X-API-Key": context.raw_credential}
    return {}
