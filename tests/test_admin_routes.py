import pytest
from unittest.mock import patch, MagicMock


# --- GET /admin/users (line 15) ---

def test_list_users_no_auth_returns_403(client):
    """Line 15: RoleChecker dependency blocks unauthenticated requests"""
    response = client.get("/admin/users")
    assert response.status_code == 403


# --- POST /admin/generate-community-reviews (lines 40-62) ---

def test_generate_community_reviews_missing_api_key(client):
    """Lines 44-49: no OPENAI_API_KEY set"""
    with patch.dict("os.environ", {}, clear=True):
        with patch("app.routes.admin.os.getenv", return_value=None):
            response = client.post("/admin/generate-community-reviews")
    assert response.status_code == 500
    assert "OPENAI_API_KEY" in response.json()["detail"]


def test_generate_community_reviews_success(client):
    """Lines 53-58: happy path"""
    mock_result = {
        "status": "success",
        "total_books_processed": 3,
        "updated": 2,
        "skipped": 1,
        "errors": []
    }
    with patch("app.routes.admin.os.getenv", return_value="fake-key"):
        with patch("app.routes.admin.SynopsisSyncService") as MockService:
            MockService.return_value.generate_all_community_reviews.return_value = mock_result
            response = client.post("/admin/generate-community-reviews")
    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_generate_community_reviews_unexpected_error(client):
    """Lines 60-62: generic exception handler"""
    with patch("app.routes.admin.os.getenv", return_value="fake-key"):
        with patch("app.routes.admin.SynopsisSyncService") as MockService:
            MockService.return_value.generate_all_community_reviews.side_effect = Exception("boom")
            response = client.post("/admin/generate-community-reviews")
    assert response.status_code == 500
    assert "Error during community review generation" in response.json()["detail"]


# --- POST /admin/sync-synopses (line 71) ---

def test_sync_synopses_delegates_to_generate(client):
    """Line 71: legacy alias calls generate_community_reviews"""
    mock_result = {"status": "success", "total_books_processed": 0, "updated": 0, "skipped": 0, "errors": []}
    with patch("app.routes.admin.os.getenv", return_value="fake-key"):
        with patch("app.routes.admin.SynopsisSyncService") as MockService:
            MockService.return_value.generate_all_community_reviews.return_value = mock_result
            response = client.post("/admin/sync-synopses")
    assert response.status_code == 200


# --- GET /admin/synopsis-moderation (lines 79-89) ---

def test_list_synopsis_moderation_success(client):
    """Lines 79-85: happy path"""
    with patch("app.routes.admin.os.getenv", return_value="fake-key"):
        with patch("app.routes.admin.SynopsisSyncService") as MockService:
            MockService.return_value.list_moderation_items.return_value = []
            response = client.get("/admin/synopsis-moderation")
    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_list_synopsis_moderation_invalid_status(client):
    """Query param pattern validation"""
    response = client.get("/admin/synopsis-moderation?status=invalid")
    assert response.status_code == 422


def test_list_synopsis_moderation_error(client):
    """Lines 87-89: exception handler"""
    with patch("app.routes.admin.os.getenv", return_value="fake-key"):
        with patch("app.routes.admin.SynopsisSyncService") as MockService:
            MockService.return_value.list_moderation_items.side_effect = Exception("db error")
            response = client.get("/admin/synopsis-moderation")
    assert response.status_code == 500


# --- POST /admin/synopsis-moderation/{id}/accept (lines 94-102) ---

def test_accept_moderation_success(client):
    """Lines 94-97: happy path"""
    with patch("app.routes.admin.os.getenv", return_value="fake-key"):
        with patch("app.routes.admin.SynopsisSyncService") as MockService:
            MockService.return_value.accept_moderation_item.return_value = {"accepted": True}
            response = client.post("/admin/synopsis-moderation/mod-123/accept")
    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_accept_moderation_not_found(client):
    """Lines 98-99: ValueError → 404"""
    with patch("app.routes.admin.os.getenv", return_value="fake-key"):
        with patch("app.routes.admin.SynopsisSyncService") as MockService:
            MockService.return_value.accept_moderation_item.side_effect = ValueError("not found")
            response = client.post("/admin/synopsis-moderation/mod-123/accept")
    assert response.status_code == 404


def test_accept_moderation_error(client):
    """Lines 100-102: generic exception → 500"""
    with patch("app.routes.admin.os.getenv", return_value="fake-key"):
        with patch("app.routes.admin.SynopsisSyncService") as MockService:
            MockService.return_value.accept_moderation_item.side_effect = Exception("unexpected")
            response = client.post("/admin/synopsis-moderation/mod-123/accept")
    assert response.status_code == 500


# --- POST /admin/synopsis-moderation/{id}/reject (lines 107-115) ---

def test_reject_moderation_success(client):
    """Lines 107-110: happy path"""
    with patch("app.routes.admin.os.getenv", return_value="fake-key"):
        with patch("app.routes.admin.SynopsisSyncService") as MockService:
            MockService.return_value.reject_moderation_item.return_value = {"rejected": True}
            response = client.post("/admin/synopsis-moderation/mod-123/reject")
    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_reject_moderation_not_found(client):
    """Lines 111-112: ValueError → 404"""
    with patch("app.routes.admin.os.getenv", return_value="fake-key"):
        with patch("app.routes.admin.SynopsisSyncService") as MockService:
            MockService.return_value.reject_moderation_item.side_effect = ValueError("not found")
            response = client.post("/admin/synopsis-moderation/mod-123/reject")
    assert response.status_code == 404


def test_reject_moderation_error(client):
    """Lines 113-115: generic exception → 500"""
    with patch("app.routes.admin.os.getenv", return_value="fake-key"):
        with patch("app.routes.admin.SynopsisSyncService") as MockService:
            MockService.return_value.reject_moderation_item.side_effect = Exception("unexpected")
            response = client.post("/admin/synopsis-moderation/mod-123/reject")
    assert response.status_code == 500