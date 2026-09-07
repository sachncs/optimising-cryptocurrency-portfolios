# syntax=docker/dockerfile:1.7

# ---------- builder ----------
FROM python:3.12-slim AS builder

ARG CPS_EXTRAS=""

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

COPY pyproject.toml README.md CHANGELOG.md LICENSE MANIFEST.in ./
COPY src ./src

RUN python -m pip install --upgrade pip build \
 && python -m build --wheel --outdir /wheels \
 && extras=$(echo "$CPS_EXTRAS" | tr -d ' '); \
    if [ -n "$extras" ]; then \
      pip install "/wheels/$(ls /wheels)"${extras:+[$extras]}; \
    else \
      pip install --no-deps /wheels/*.whl; \
    fi

# ---------- runtime ----------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CPS_OUTPUT_DIR=/data/outputs \
    CPS_RUN_DIR=/data/runs

RUN groupadd --system --gid 10001 cps \
 && useradd  --system --uid 10001 --gid cps --home-dir /home/cps --shell /sbin/nologin cps \
 && mkdir -p /data/outputs /data/runs \
 && chown -R cps:cps /data /home/cps

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

USER cps
WORKDIR /home/cps

VOLUME ["/data/outputs", "/data/runs"]

ENTRYPOINT ["crypto-portfolio"]
CMD ["--output-dir", "/data/outputs", "--run-dir", "/data/runs"]