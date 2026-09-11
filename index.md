---
layout: default
title: Crypto Portfolio System
permalink: /
---

# Crypto Portfolio System

Consensus-clustered cryptocurrency portfolio construction, backtesting,
and serving — built on the framework described in
[arXiv:2505.24831v2](https://arxiv.org/abs/2505.24831v2).

![Pipeline diagram]({{ '/architecture.svg' | relative_url }})

## What it does

The [`crypto-portfolio-system`](https://pypi.org/project/crypto-portfolio-system/)
Python package ingests price data, forecasts returns, builds rolling
correlation networks, extracts stable asset clusters via consensus
Louvain community detection, then performs Sharpe-ratio portfolio
optimisation with covariance regularisation, risk limits, and
execution costs.

It ships with:

- a CLI (`crypto-portfolio` and `cps-realtime`),
- a stateless FastAPI REST surface,
- a production-ready multi-stage Docker image.

## Quick Start

```bash
pip install crypto-portfolio-system
crypto-portfolio --output-dir outputs --run-dir runs
```

That produces `outputs/trades.csv`, `outputs/summary.csv`,
`outputs/log_returns.csv`, and a matching `events.jsonl` plus
`metrics.json` on disk.

## Repo and Paper

- [GitHub repository](https://github.com/sachncs/optimising-cryptocurrency-portfolios)
- [arXiv paper](https://arxiv.org/abs/2505.24831v2)
- [PyPI release](https://pypi.org/project/crypto-portfolio-system/)
