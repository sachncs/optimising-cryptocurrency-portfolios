from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("starlette")

from starlette.testclient import TestClient  # noqa: E402

from cps.api import create_app  # noqa: E402


@pytest.fixture()
def client(tmp_path):
    app = create_app(tmp_path / "cps_data")
    return TestClient(app)


@pytest.fixture()
def price_rows():
    rows = [["date", "btc", "eth", "sol"]]
    for i in range(20):
        rows.append([f"2024-01-{i + 1:02d}", 100.0 + i, 50.0 + 0.5 * i, 10.0 + 0.5 * i])
    return rows


def _base_config():
    return {
        "train_window_days": 5,
        "correlation_window_days": 3,
        "rebalance_step_days": 2,
        "horizons_days": [1],
        "consensus_runs": 2,
        "forecast_method": "naive",
        "max_volatility_annual": 5.0,
    }


def test_health(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "base_dir" in payload


def test_create_run_with_inline_prices(client, price_rows):
    body = {"config": _base_config(), "prices": price_rows}
    response = client.post("/api/v1/runs", json=body)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["trades_count"] >= 0
    assert "run_id" in payload
    assert "trades_json" in payload["artifact_paths"]


def test_create_run_rejects_missing_prices(client):
    response = client.post("/api/v1/runs", json={"config": {}})
    assert response.status_code == 400


def test_get_run_returns_artifacts(client, price_rows):
    body = {"config": _base_config(), "prices": price_rows}
    run_id = client.post("/api/v1/runs", json=body).json()["run_id"]
    response = client.get(f"/api/v1/runs/{run_id}")
    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == run_id
    assert "artifacts" in payload


def test_get_run_404_when_missing(client):
    response = client.get("/api/v1/runs/no-such-run")
    assert response.status_code == 404


def test_summary_and_trades_endpoints(client, price_rows):
    body = {"config": _base_config(), "prices": price_rows}
    run_id = client.post("/api/v1/runs", json=body).json()["run_id"]
    summary = client.get(f"/api/v1/runs/{run_id}/summary").json()
    trades = client.get(f"/api/v1/runs/{run_id}/trades?limit=5").json()
    metrics = client.get(f"/api/v1/runs/{run_id}/metrics").json()
    returns = client.get(f"/api/v1/runs/{run_id}/log-returns?max_rows=3").json()
    assert isinstance(summary["summary"], list)
    assert "limit" in trades
    assert "counters" in metrics["metrics"]
    assert returns["total_rows"] >= 1


def test_create_run_with_csv_content(client):
    csv_text = (
        "date,a,b,c\n"
        "2024-01-01,10,20,30\n"
        "2024-01-02,11,21,29\n"
        "2024-01-03,12,20,28\n"
        "2024-01-04,13,22,27\n"
        "2024-01-05,14,23,26\n"
        "2024-01-06,15,24,25\n"
        "2024-01-07,16,25,24\n"
        "2024-01-08,17,26,23\n"
        "2024-01-09,18,27,22\n"
        "2024-01-10,19,28,21\n"
    )
    config = dict(_base_config())
    config["train_window_days"] = 4
    config["correlation_window_days"] = 2
    config["rebalance_step_days"] = 1
    body = {"config": config, "prices_csv_content": csv_text}
    response = client.post("/api/v1/runs", json=body)
    assert response.status_code == 200


def test_create_run_with_csv_path(tmp_path, client, price_rows):
    csv_path = tmp_path / "prices.csv"
    with csv_path.open("w") as fh:
        for row in price_rows:
            fh.write(",".join(str(v) for v in row) + "\n")
    body = {"config": _base_config(), "prices_csv_path": str(csv_path)}
    response = client.post("/api/v1/runs", json=body)
    assert response.status_code == 200


def test_invalid_inline_prices_payload(client):
    response = client.post("/api/v1/runs", json={"config": {}, "prices": []})
    assert response.status_code == 400


def test_prices_with_missing_date_column(client):
    response = client.post("/api/v1/runs", json={"config": {}, "prices": [["x", "y"], [1, 2]]})
    assert response.status_code == 400


def test_create_run_unknown_forecast_method(client, price_rows):
    body = {"config": _base_config() | {"forecast_method": "bogus"}, "prices": price_rows}
    response = client.post("/api/v1/runs", json=body)
    assert response.status_code == 400
