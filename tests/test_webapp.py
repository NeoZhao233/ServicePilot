import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from servicepilot.webapp import create_app

ROOT = Path(__file__).resolve().parents[1]


class WebAppTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(create_app(data_dir=ROOT / "data"))

    def test_health_frontend_and_openapi_are_served(self) -> None:
        self.assertEqual(self.client.get("/api/health").json(), {"status": "ok"})

        page = self.client.get("/").text
        self.assertIn("ServicePilot", page)
        self.assertIn("Agent Trace", page)

        schema = self.client.get("/openapi.json").json()
        self.assertEqual(schema["info"]["title"], "ServicePilot API")
        self.assertIn("/api/chat", schema["paths"])

    def test_chat_endpoint_returns_trace(self) -> None:
        payload = self.client.post(
            "/api/chat",
            json={"user_id": "USER-001", "message": "ORD-1003 的物流到哪里了"},
        ).json()
        self.assertIn("运输中", payload["message"])
        self.assertEqual(payload["intent"], "logistics")
        self.assertTrue(any(event["name"] == "get_order" for event in payload["events"]))

        sessions = self.client.get("/api/sessions", params={"user_id": "USER-001"}).json()[
            "sessions"
        ]
        self.assertEqual(sessions[0]["session_id"], payload["session_id"])

        detail = self.client.get(
            f"/api/sessions/{payload['session_id']}", params={"user_id": "USER-001"}
        ).json()
        self.assertEqual(detail["turn_count"], 1)
        self.assertTrue(any(event["name"] == "get_order" for event in detail["events"]))

    def test_validation_and_session_authorization(self) -> None:
        self.assertEqual(self.client.post("/api/chat", json={}).status_code, 422)
        response = self.client.post(
            "/api/chat",
            json={"user_id": "USER-001", "message": "ORD-1003 的物流到哪里了"},
        ).json()
        denied = self.client.get(
            f"/api/sessions/{response['session_id']}", params={"user_id": "USER-002"}
        )
        self.assertEqual(denied.status_code, 403)


if __name__ == "__main__":
    unittest.main()


def test_authenticated_identity_cannot_be_selected_by_request(monkeypatch):
    monkeypatch.setenv("SERVICEPILOT_AUTH_TOKENS", '{"local-test-token":"USER-001"}')
    client = TestClient(create_app(data_dir=ROOT / "data"))
    payload = {"user_id": "USER-001", "message": "ORD-1001 物流"}
    assert client.post("/api/chat", json=payload).status_code == 401
    headers = {"Authorization": "Bearer local-test-token"}
    assert client.post("/api/chat", headers=headers, json=payload).status_code == 200
    payload["user_id"] = "USER-002"
    assert client.post("/api/chat", headers=headers, json=payload).status_code == 403
