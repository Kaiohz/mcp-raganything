"""Tests for FastAPI lifespan management."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def main_middleware_singleton():
    """The single wired ``McpApiKeyMiddleware`` instance owned by ``main``."""
    import main

    return main.mcp_middleware


class TestLifespan:
    """Tests for lifespan context managers in main.py."""

    @pytest.mark.asyncio
    async def test_db_lifespan_closes_classical_bm25_pool_on_shutdown(self):
        """Should close Classical BM25 adapter connection pool on shutdown."""
        from main import db_lifespan

        mock_app = MagicMock()
        mock_bm25 = AsyncMock()

        with (
            patch("main.classical_bm25_adapter", mock_bm25),
            patch("main.classical_vector_store", None),
        ):
            async with db_lifespan(mock_app):
                pass
            mock_bm25.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_db_lifespan_handles_no_classical_bm25_adapter(self):
        """Should handle gracefully when classical_bm25_adapter is None."""
        from main import db_lifespan

        mock_app = MagicMock()

        with (
            patch("main.classical_bm25_adapter", None),
            patch("main.classical_vector_store", None),
        ):
            async with db_lifespan(mock_app):
                pass

    @pytest.mark.asyncio
    async def test_db_lifespan_handles_close_failure(self):
        """Should not crash if Classical BM25 close fails."""
        from main import db_lifespan

        mock_app = MagicMock()
        mock_bm25 = AsyncMock()
        mock_bm25.close = AsyncMock(side_effect=Exception("Close failed"))

        with (
            patch("main.classical_bm25_adapter", mock_bm25),
            patch("main.classical_vector_store", None),
        ):
            async with db_lifespan(mock_app):
                pass
            mock_bm25.close.assert_called_once()

    def test_run_fastapi_starts_uvicorn(self):
        """Should start uvicorn when run_fastapi is called."""
        with patch("main.uvicorn.run") as mock_uvicorn:
            from main import run_fastapi

            run_fastapi()
            mock_uvicorn.assert_called_once()

    def test_run_fastapi_serves_the_in_process_app_object(self):
        """BUG-005 regression: uvicorn must serve the already-built app object.

        The container entrypoint executes ``main`` as ``__main__``; serving an
        import string would make uvicorn execute the module a second time in
        the same process, stacking an unwired ``McpApiKeyMiddleware`` on the
        shared FastMCP servers in front of the wired guard.
        """
        from fastapi import FastAPI

        import main

        with patch("main.uvicorn.run") as mock_uvicorn:
            main.run_fastapi()

        assert mock_uvicorn.call_count == 1
        served_app = mock_uvicorn.call_args.args[0]
        # The served object is THE in-process application, not an import
        # string: one module execution, one middleware per mount, one lifespan
        # wiring it (BUG-005).
        assert isinstance(served_app, FastAPI)
        assert served_app is main.app

        call_kwargs = mock_uvicorn.call_args.kwargs
        assert call_kwargs["host"] == main.app_config.HOST
        assert call_kwargs["port"] == main.app_config.PORT
        assert call_kwargs["ws"] == "none"
        # No module import-string is passed anywhere (the historical trigger).
        assert all(not isinstance(arg, str) for arg in mock_uvicorn.call_args.args)

    def test_mcp_servers_carry_exactly_one_wired_guard(self):
        """Each shared FastMCP server carries exactly one McpApiKeyMiddleware.

        ``main`` must be executed only once per process: a second execution of
        the module would append a second — unwired — guard to these same
        singletons (``FastMCP.add_middleware`` only appends), and with a
        non-empty legacy master key that unwired guard rejects legitimate
        credentials (BUG-005). FastMCP also seeds its own built-ins, so count
        only our guard type.
        """
        from application.api.mcp_classical_tools import mcp_classical
        from application.api.mcp_file_tools import mcp_files
        from application.api.mcp_web_tools import mcp_web
        from security import McpApiKeyMiddleware

        our_guard = main_middleware_singleton()
        for server in (mcp_files, mcp_classical, mcp_web):
            guards = [
                mw for mw in server.middleware if isinstance(mw, McpApiKeyMiddleware)
            ]
            assert guards == [our_guard], "unexpected guard duplication"
