"""Automated tests for FastAPI endpoints using httpx AsyncClient and direct lifespan context."""

import pytest
from httpx import AsyncClient, ASGITransport
from api.main import app, lifespan


@pytest.mark.anyio
async def test_health_check():
    """Verify /health returns 200 and confirms the model is loaded in memory."""
    async with lifespan(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


@pytest.mark.anyio
async def test_model_info():
    """Verify /model-info returns the classifier and preprocessor names."""
    async with lifespan(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_type"] == "XGBClassifier"
    assert data["preprocessor"] == "ColumnTransformer"


@pytest.mark.anyio
async def test_predict_low_and_high_risk_orders():
    """Test /predict with different order inputs (short-distance vs cross-country)."""
    low_risk_payload = {
        "price": 120.00,
        "freight_value": 10.00,
        "product_weight_g": 300.0,
        "product_length_cm": 16.0,
        "product_height_cm": 10.0,
        "product_width_cm": 12.0,
        "seller_customer_distance_km": 25.0,
        "customer_state": "SP",
        "seller_state": "SP",
        "order_purchase_timestamp": "2018-05-10 14:30:00",
    }

    high_risk_payload = {
        "price": 45.00,
        "freight_value": 65.00,
        "product_weight_g": 8500.0,
        "product_length_cm": 60.0,
        "product_height_cm": 50.0,
        "product_width_cm": 40.0,
        "seller_customer_distance_km": 2800.0,
        "customer_state": "MA",
        "seller_state": "SP",
        "order_purchase_timestamp": "2017-11-24 22:15:00",
    }

    async with lifespan(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            res_low = await ac.post("/predict", json=low_risk_payload)
            res_high = await ac.post("/predict", json=high_risk_payload)

    assert res_low.status_code == 200
    assert res_high.status_code == 200

    low_data = res_low.json()
    high_data = res_high.json()

    assert 0.0 <= low_data["delay_probability"] <= 1.0
    assert 0.0 <= high_data["delay_probability"] <= 1.0
    assert low_data["risk_level"] in ["Low Risk", "High Risk"]
    assert high_data["risk_level"] in ["Low Risk", "High Risk"]
    # A heavy cross-country order to MA during Black Friday should have higher delay probability than local SP
    assert high_data["delay_probability"] > low_data["delay_probability"]


@pytest.mark.anyio
async def test_predict_batch():
    """Test /predict/batch endpoint with multiple orders."""
    batch_payload = [
        {
            "price": 99.90,
            "freight_value": 15.50,
            "product_weight_g": 1200.0,
            "product_length_cm": 30.0,
            "product_height_cm": 15.0,
            "product_width_cm": 20.0,
            "seller_customer_distance_km": 450.5,
            "customer_state": "SP",
            "seller_state": "RJ",
            "order_purchase_timestamp": "2017-10-02 10:00:00",
        },
        {
            "price": 250.00,
            "freight_value": 45.00,
            "product_weight_g": 4500.0,
            "product_length_cm": 40.0,
            "product_height_cm": 30.0,
            "product_width_cm": 25.0,
            "seller_customer_distance_km": 1800.0,
            "customer_state": "BA",
            "seller_state": "PR",
            "order_purchase_timestamp": "2018-03-15 18:20:00",
        },
    ]

    async with lifespan(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post("/predict/batch", json=batch_payload)

    assert response.status_code == 200
    results = response.json()
    assert isinstance(results, list)
    assert len(results) == 2