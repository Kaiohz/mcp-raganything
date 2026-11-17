"""Adapter-local ``httpx`` seam.

The memory-store adapter obtains its ``AsyncClient`` exclusively through this
module so outbound HTTP can be replaced per-adapter without touching the
global ``httpx`` namespace.
"""

from httpx import AsyncClient, HTTPError, Response, TimeoutException

__all__ = ["AsyncClient", "HTTPError", "Response", "TimeoutException"]
