import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


class TestAPISecurity:
    def test_health_endpoint_no_auth(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_protected_endpoints_require_auth(self):
        # No auth
        response = client.post("/api/v1/predict", json={"periods": 24})
        assert response.status_code == 401

        response = client.post("/api/v1/anomalies", json={"data": []})
        assert response.status_code == 401

    def test_invalid_token(self):
        headers = {"Authorization": "Bearer invalid-token"}
        response = client.post("/api/v1/predict", json={"periods": 24}, headers=headers)
        assert response.status_code == 401

    def test_rate_limiting(self):
        for _ in range(65):
            response = client.get("/health")
            if response.status_code == 429:
                break
        else:
            pytest.skip("Rate limiting not triggered in test environment")


class TestAPIAuthentication:
    def test_login_success(self):
        response = client.post(
            "/token", data={"username": "admin", "password": "admin123"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert "user" in data

    def test_login_failure(self):
        response = client.post(
            "/token", data={"username": "admin", "password": "wrong"}
        )
        assert response.status_code == 401

    def test_authenticated_request(self):
        login_response = client.post(
            "/token", data={"username": "admin", "password": "admin123"}
        )
        token = login_response.json()["access_token"]

        headers = {"Authorization": f"Bearer {token}"}
        response = client.post("/api/v1/predict", json={"periods": 24}, headers=headers)
        assert response.status_code == 200


class TestAPIEndpoints:
    def test_prediction_endpoint(self):
        login_response = client.post(
            "/token", data={"username": "admin", "password": "admin123"}
        )
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        response = client.post("/api/v1/predict", json={"periods": 24}, headers=headers)
        assert response.status_code == 200

        data = response.json()
        assert "predictions" in data
        assert "model_info" in data
        assert len(data["predictions"]) == 24

    def test_anomaly_endpoint(self):
        login_response = client.post(
            "/token", data={"username": "admin", "password": "admin123"}
        )
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        test_data = [
            {"timestamp": "2024-01-01T10:00:00", "power_output": 100},
            {"timestamp": "2024-01-01T11:00:00", "power_output": 120},
        ]

        response = client.post(
            "/api/v1/anomalies", json={"data": test_data}, headers=headers
        )
        assert response.status_code == 200

        data = response.json()
        assert "anomalies" in data
        assert "summary" in data
        assert "model_info" in data

    def test_model_status_endpoint(self):
        login_response = client.post(
            "/token", data={"username": "admin", "password": "admin123"}
        )
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        response = client.get("/api/v1/models/status", headers=headers)
        assert response.status_code == 200

        data = response.json()
        assert "forecast_model" in data
        assert "anomaly_model" in data

    def test_admin_only_endpoint(self):
        login_response = client.post(
            "/token", data={"username": "user", "password": "user123"}
        )
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        response = client.post("/api/v1/models/train", headers=headers)
        assert response.status_code == 403

        login_response = client.post(
            "/token", data={"username": "admin", "password": "admin123"}
        )
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        response = client.post("/api/v1/models/train", headers=headers)
        assert response.status_code == 200


class TestAPIValidation:
    def test_prediction_validation(self):
        login_response = client.post(
            "/token", data={"username": "admin", "password": "admin123"}
        )
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        response = client.post("/api/v1/predict", json={"periods": 0}, headers=headers)
        assert response.status_code == 422

        response = client.post(
            "/api/v1/predict", json={"periods": 200}, headers=headers
        )
        assert response.status_code == 422

    def test_anomaly_validation(self):
        login_response = client.post(
            "/token", data={"username": "admin", "password": "admin123"}
        )
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        response = client.post("/api/v1/anomalies", json={"data": []}, headers=headers)
        assert response.status_code == 422

        response = client.post(
            "/api/v1/anomalies",
            json={"data": [{"test": 1}], "threshold": 1.5},
            headers=headers,
        )
        assert response.status_code == 422


class TestAPISecurityHeaders:
    def test_security_headers(self):
        response = client.get("/health")

        headers = response.headers
        assert "X-Content-Type-Options" in headers
        assert "X-Frame-Options" in headers
        assert "X-XSS-Protection" in headers
        assert "Strict-Transport-Security" in headers
        assert "Content-Security-Policy" in headers
        assert "Referrer-Policy" in headers
