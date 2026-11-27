"""Shared domain constants."""

DEFAULT_WORKING_DIR = "default-composable"

# --- Caller scope (user × agent, scope-rag-memories spec) ---------------------
#
# The agent identity travels in the ``X-Composable-Agent`` header injected by
# the platform; absent → the ``personal`` scope. The charset rule (REQ-002)
# is shared by the scope resolver and the memory-name validator.
AGENT_HEADER = "x-composable-agent"
PERSONAL_AGENT_SLUG = "personal"
AGENT_SLUG_PATTERN = r"^[a-z0-9][a-z0-9-]{0,99}$"
#: Conservative cap on LIVE RAG spaces per user (agents + subagents + external
#: coding scopes). Exceeding it on ``POST /api/v1/spaces`` → 422 (security
#: review FINDING-4: no per-user cap lets an authenticated caller generate
#: unbounded ``rag_spaces`` rows). 256 comfortably exceeds realistic platform
#: usage while bounding the abuse surface.
MAX_SPACES_PER_USER = 256
