"""validate-with-id merge rule (UI BUG-001): stored entry ⊕ dirty payload.

Pure domain rule: given the STORED registry entry (secrets decrypted by the
store on read; RLS-scoped to the caller) and the client's dirty validate
payload values, return the EFFECTIVE connection values the ``POST /validate``
probe must run with.

The merge contract (feat-mcp-oauth2-dual-auth BUG-001):

- the masked UI (REQ-016 / REQ-018) cannot send the stored secrets back, so
  ONLY a real, non-empty override value may replace a stored one — absent
  and masked-empty values (``headers={}``, ``env={}``, ``auth_token=""``,
  ``auth_token=None``) NEVER clear a stored secret ("Test connection" must
  not destroy stored state by omission);
- ``auth_mode`` (REQ-004, non-secret metadata) follows the current validate
  semantics: the explicitly provided request value wins over the stored
  mode; a withheld one takes the stored mode (or the legacy default when
  the probe validates a fresh candidate without an ``id``).

The connection-dict assembly (placeholder resolution, injected-credential
merge order) stays at the transport boundary — this rule only resolves the
per-field effective VALUES.
"""

from typing import NamedTuple

from domain.entities.mcp_server_config import McpAuthMode
from domain.entities.mcp_server_registry_entry import RegisteredMcpServer


class McpValidateOverrides(NamedTuple):
    """The client's dirty validate payload values.

    Each member is the override the payload carries for one field; ``None``
    means "absent or masked-empty" (the field carries NO override — the
    stored value, if any, wins). Built at the transport boundary (the
    validate route), consumed by :func:`resolve_validate_candidate`.
    """

    url: str | None
    headers: dict[str, str] | None
    env: dict[str, str] | None
    auth_token: str | None
    auth_mode: McpAuthMode | None
    source_type: str | None
    openapi_url: str | None


class McpValidateCandidateSpec(NamedTuple):
    """The effective field values for one ``POST /validate`` probe.

    Attributes:
        url: Remote URL to probe (``None`` only for an ``openapi`` source).
        headers: HTTP headers the probe sends to the target.
        env: Environment values (unused for http probes, kept for parity).
        auth_token: Bearer token the probe transmits (``None`` = unset).
        auth_mode: Credential-transmission mode of the probe (REQ-004).
        source_type: ``"external"`` (network probe) or ``"openapi"``.
        openapi_url: Spec URL of the probe (``openapi`` source only).
    """

    url: str | None
    headers: dict[str, str]
    env: dict[str, str]
    auth_token: str | None
    auth_mode: McpAuthMode
    source_type: str
    openapi_url: str | None


def _pick_value[T](override: T | None, fallback: T) -> T:
    """Pick the effective scalar value: an explicit truthy override wins.

    Enum members and non-secret mode strings are always truthy, so
    truthiness and "explicitly provided" coincide for every scalar field.

    Args:
        override: The client's dirty value (``None``/empty = masked).
        fallback: The value the stored entry (or the legacy candidate
            defaults) supplies.

    Returns:
        The effective value for the probe.
    """
    return override if override else fallback


def _pick_mapping(
    override: dict[str, str] | None, fallback: dict[str, str]
) -> dict[str, str]:
    """Pick the effective mapping, copying the winner.

    Copies (never aliases) the selected mapping so a probe consumer cannot
    mutate the frozen stored entity or this module's legacy defaults.

    Args:
        override: The client's dirty mapping (``None``/empty = masked).
        fallback: The stored (or legacy-default) mapping.

    Returns:
        An independent copy of the effective mapping.
    """
    return dict(override) if override else dict(fallback)


#: The legacy stateless candidate defaults (pre-``id`` validate semantics):
#: projected from whenever there is no stored entry to merge over.
_LEGACY_CANDIDATE_DEFAULTS = McpValidateCandidateSpec(
    url=None,
    headers={},
    env={},
    auth_token=None,
    auth_mode=McpAuthMode.HEADERS,
    source_type="external",
    openapi_url=None,
)


def resolve_validate_candidate(
    stored: RegisteredMcpServer | None,
    overrides: McpValidateOverrides,
) -> McpValidateCandidateSpec:
    """Resolve the effective values for one ``POST /validate`` probe.

    Args:
        stored: The RLS-scoped stored entry supplying the defaults (``None``
            = plain candidate validation: every masked field takes its
            legacy default, i.e. today's stateless semantics).
        overrides: The client's dirty payload values (``None`` = absent or
            masked-empty; a real value REPLACES the stored one).

    Returns:
        The effective probe specification. Stored secrets never flow back to
        the client: the route answers a tool count only.
    """
    origin: RegisteredMcpServer | McpValidateCandidateSpec = (
        stored if stored is not None else _LEGACY_CANDIDATE_DEFAULTS
    )
    return McpValidateCandidateSpec(
        url=_pick_value(overrides.url, origin.url),
        headers=_pick_mapping(overrides.headers, origin.headers),
        env=_pick_mapping(overrides.env, origin.env),
        auth_token=_pick_value(overrides.auth_token, origin.auth_token),
        auth_mode=_pick_value(overrides.auth_mode, origin.auth_mode),
        source_type=_pick_value(overrides.source_type, origin.source_type),
        openapi_url=_pick_value(overrides.openapi_url, origin.openapi_url),
    )
