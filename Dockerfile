# Multi-stage build for MCP service
# Stage 1: Build dependencies with uv
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
RUN uv sync --frozen --no-dev --no-cache

# Stage 2: Runtime image
FROM python:3.13-slim-bookworm

# Install only critical runtime system deps, then clean up apt metadata to keep image slim.
# libatomic1 is required by the Playwright node driver itself (the slim Debian
# image does not ship it — without it the driver dies instantly and the
# browser-installation step never runs, BUG-001). The remaining libs are the
# headless-Chromium runtime set for bookworm.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-fra \
        libatomic1 libnss3 libnspr4 libasound2 libatk1.0-0 libatk-bridge2.0-0 \
        libatspi2.0-0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 \
        libxkbcommon0 libgbm1 libpango-1.0-0 libcairo2 \
    && apt-get upgrade -y libgnutls30 libssh2-1 \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Non-root runtime identity (created early so the browser-install stage below
# never needs login shells). /home/appuser stays writable so the pinned
# Crawl4AI/Playwright tools can lazily write their caches/logs at runtime;
# browsers themselves are never downloaded again after build time.
RUN useradd -m -u 1000 appuser

# Set TESSDATA_PREFIX for Tesseract and create Kreuzberg cache symlink
# Kreuzberg hardcodes linux-aarch64 path, so we symlink to the actual tessdata location
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata
RUN mkdir -p /io/.tesseract-cache/linux-aarch64 \
    && ln -s ${TESSDATA_PREFIX} /io/.tesseract-cache/linux-aarch64/tessdata

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Crawl4AI browser support: install Chromium + system dependencies at build
# time (as root), exactly once, into a stable location readable by appuser.
# PLAYWRIGHT_BROWSERS_PATH keeps browsers out of HOME; no browser download
# happens at runtime (and definitely not as root). Crawl4AI's own base/cache
# directory (CRAWL4_AI_BASE_DIRECTORY) points at the non-root user's HOME so
# runtime writes (crawler.log, crawl4ai db) land on writable storage.
ENV PYTHONPATH=/app/src \
    PATH="/app/.venv/bin:$PATH" \
    PLAYWRIGHT_BROWSERS_PATH=/opt/crawl4ai-browsers \
    CRAWL4_AI_BASE_DIRECTORY=/home/appuser \
    HOME=/home/appuser

# Step 1 — install the Chromium build that MATCHES the locked Playwright
# (`python -m playwright install --with-deps --force chromium` run directly so
# a failed install FAILS THE BUILD: crawl4ai-setup delegates to the same
# command but swallows non-zero exits via crawl4ai.install.install_playwright,
# which only warns — that swallowing is what previously left the image with a
# mismatched browser build, BUG-001). `--with-deps` also apt-installs any
# remaining Playwright system dependencies as root.
RUN python -m playwright install --with-deps --force chromium \
    && chmod -R a+rX /opt/crawl4ai-browsers

# Step 2 — crawl4ai's own post-install setup (patchright browsers for
# undetected mode + crawl4ai db init). Its playwright step is now a no-op
# (chromium 1255 already installed), so the swallowed-failure path it exists
# to recover from cannot occur here.
RUN crawl4ai-setup \
    && chmod -R a+rX /opt/crawl4ai-browsers \
    && mkdir -p /home/appuser/.cache /home/appuser/.crawl4ai \
    && chown -R appuser:appuser /home/appuser

# Copy application source
COPY src/ /app/src/
COPY .env.example /app/.env

# Set Python path to include src directory (kept in sync with the ENV above)
ENV PYTHONPATH=/app/src
ENV PATH="/app/.venv/bin:$PATH"

# Give the non-root user ownership of the application tree
RUN chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Default command: run FastAPI server
CMD ["python","src/main.py"]
