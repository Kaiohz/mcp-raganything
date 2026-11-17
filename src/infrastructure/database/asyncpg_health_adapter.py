import logging
import ssl as ssl_module

import asyncpg

from config import DatabaseConfig
from domain.logging.messages import LogMessage
from domain.ports.postgres_health_port import PostgresHealthPort

logger = logging.getLogger(__name__)

_SSL_MODES_NO_VERIFY = frozenset({"require", "prefer"})


class AsyncpgHealthAdapter(PostgresHealthPort):
    """PostgreSQL health check using asyncpg direct connection (port-backed)."""

    def __init__(self, db_config: DatabaseConfig) -> None:
        self._db_url = db_config.asyncpg_url
        self._statement_cache_size = db_config.POSTGRES_STATEMENT_CACHE_SIZE
        self._ssl_mode = db_config.ssl_mode

    async def ping(self) -> bool:
        try:
            connect_kwargs: dict = {}
            if self._statement_cache_size is not None:
                connect_kwargs["statement_cache_size"] = self._statement_cache_size
            ctx = self._build_ssl_context()
            if ctx is not None:
                connect_kwargs["ssl"] = ctx
            conn = await asyncpg.connect(self._db_url, **connect_kwargs)
            try:
                await conn.fetchval("SELECT 1")
                return True
            finally:
                await conn.close()
        except Exception:
            logger.warning(LogMessage.POSTGRES_HEALTH_CHECK_FAILED, exc_info=True)
            return False

    def _build_ssl_context(self) -> ssl_module.SSLContext | None:
        """Build an SSL context appropriate for the configured sslmode.

        - ``require`` / ``prefer``: PostgreSQL's documented "encryption without
          certificate or hostname verification" mode (CERT_NONE, check_hostname
          disabled). Intentionally supported only for these operator-selected
          modes; a warning is logged on every use.
        - ``verify-ca`` / ``verify-full`` (and any unknown sslmode): strict
          default context — certificate chain AND server hostname are verified
          via ``create_default_context``. python's ``ssl`` module cannot express
          PostgreSQL's ``verify-ca`` ("chain only, no hostname matching")
          without disabling hostname validation (python:S5527); a hostname
          mismatch therefore fails this health probe (``ping`` returns False)
          instead of being accepted — use ``require``/``prefer`` if unverified
          encryption is deliberately wanted.
        - ``disable`` or ``None``: let asyncpg handle the connection natively,
          without a custom SSL context.
        """
        if not self._ssl_mode or self._ssl_mode == "disable":
            return None

        if self._ssl_mode in _SSL_MODES_NO_VERIFY:
            logger.warning(
                LogMessage.POSTGRES_HEALTH_TLS_NO_VERIFY,
                self._ssl_mode,
            )
            # NOSONAR — sslmode=require/prefer selects PostgreSQL's documented
            # "encryption WITHOUT certificate/hostname verification" mode; the
            # constructs below are that operator-selected mode implemented
            # exactly, and a warning is logged on every use.
            ctx = ssl_module.create_default_context()  # NOSONAR — sslmode=require/prefer (PostgreSQL spec): operator-selected unverified encryption, warning logged above
            ctx.check_hostname = False  # NOSONAR — sslmode=require/prefer (PostgreSQL spec): operator-selected unverified encryption, warning logged above
            ctx.verify_mode = ssl_module.CERT_NONE  # NOSONAR — sslmode=require/prefer (PostgreSQL spec): operator-selected unverified encryption, warning logged above
            return ctx

        # verify-ca / verify-full / unknown sslmode: strict default context —
        # the certificate chain AND the server hostname must match (python
        # ssl cannot express ``verify-ca``'s "chain-only, no hostname" without
        # disabling hostname validation).
        return ssl_module.create_default_context()
