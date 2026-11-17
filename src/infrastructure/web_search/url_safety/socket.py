"""Adapter-local ``socket`` seam.

The URL-safety adapter resolves DNS exclusively through this module so the
stdlib ``getaddrinfo`` entry point stays patchable per-adapter without
touching the global ``socket`` module.
"""

from socket import getaddrinfo

__all__ = ["getaddrinfo"]
