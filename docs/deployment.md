# Deployment Guide

## Local Deployment

### Development

```bash
pip install -e ".[dev]"
crypto-portfolio --output-dir outputs --run-dir runs
```

### Production

```bash
pip install .
crypto-portfolio \
  --output-dir /var/data/outputs \
  --run-dir /var/data/runs \
  --forecast-method arima \
  --seed 42
```

## Docker

A production multi-stage image is provided at the repository root.

### Build

```bash
docker build -t crypto-portfolio-system:0.1.0 .
```

The build stage compiles a wheel and installs it into a clean `python:3.12-slim`
runtime image. The image runs as a non-root user (`uid=10001`, `cps`).

### Run (synthetic data, default)

```bash
docker run --rm crypto-portfolio-system:0.1.0
```

### Run with persistent outputs

```bash
docker run --rm \
  -v "$PWD/outputs:/data/outputs" \
  -v "$PWD/runs:/data/runs" \
  crypto-portfolio-system:0.1.0
```

### Run with custom CSV input

```bash
docker run --rm \
  -v "$PWD/prices.csv:/home/cps/prices.csv:ro" \
  -v "$PWD/outputs:/data/outputs" \
  -v "$PWD/runs:/data/runs" \
  crypto-portfolio-system:0.1.0 \
  --prices-csv /home/cps/prices.csv --date-col date
```

### Inspect the image

```bash
docker run --rm --entrypoint python crypto-portfolio-system:0.1.0 -m pip list
docker run --rm --entrypoint crypto-portfolio crypto-portfolio-system:0.1.0 --help
```

### Image details

- Base: `python:3.12-slim`
- Entrypoint: `crypto-portfolio`
- Working directory: `/home/cps`
- Volume mounts: `/data/outputs`, `/data/runs`
- User: non-root `cps` (uid 10001)
- Optional extras (e.g. `forecast-lstm`, `api`, `realtime`) are **not** installed in
  the default image to keep it small. Build a variant image to include them:

  ```bash
  docker build --build-arg CPS_EXTRAS="forecast-lstm api" -t crypto-portfolio-system:all .
  ```

## REST API

The FastAPI surface (`cps.api`) is stateless. Run it with uvicorn:

```bash
pip install -e ".[api]"
uvicorn cps.api:create_app --factory --host 0.0.0.0 --port 8000
```

In production, run multiple uvicorn workers behind a reverse proxy:

```bash
uvicorn cps.api:create_app --factory --host 0.0.0.0 --port 8000 --workers 4
```

Mount a shared volume at the `CPS_API_BASE_DIR` (default `./cps_data`) so
replicas can read previously written artifacts. The app holds no in-process
state.

## Real-time Poller

The `cps-realtime` console script runs a ccxt polling loop. Launch it as a
sidecar / long-running service via cron, systemd, or your scheduler of
choice:

```bash
pip install -e ".[realtime]"
cps-realtime --exchange binance --symbols BTC/USDT,ETH/USDT \
  --output-csv /data/rt_prices.csv --timeframe 1m --interval-seconds 60 \
  --max-iterations 1440
```

## Environment Considerations

- Ensure sufficient memory for large correlation matrices
- ARIMA forecasting is CPU-intensive; consider hardware resources
- Output directories need write permissions
- Run directories should persist across executions for idempotency

## Monitoring

The system outputs structured events and metrics:

- **events.jsonl** - Real-time pipeline events for monitoring
- **metrics.json** - Counters and timing for performance tracking

## Scheduling

For periodic rebalancing, use cron or a workflow scheduler:

```bash
# Example: run weekly
0 0 * * 0 cd /path/to/project && crypto-portfolio --output-dir outputs --run-dir runs
```
