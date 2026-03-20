import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from fastapi import HTTPException

# Patch Cognito-related side effects BEFORE importing app
with patch("app.services.cognito_service.boto3.client") as mock_boto_client, \
     patch("app.services.cognito_service.CognitoService._get_cognito_jwks", return_value={"keys": []}):

    mock_boto_client.return_value = MagicMock()
    from app.main import app

from app.dependencies.auth import get_current_user
from app.routes.bookshelf import _extract_user_id


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_dependency_overrides():
    app.dependency_overrides = {}
    yield
    app.dependency_overrides = {}


@pytest.fixture
def mock_bookshelf_service():
    return MagicMock()


@pytest.fixture
def mock_user_obj():
    user = MagicMock()
    user.user_id = "user-123"
    user.sub = None
    user.id = None
    return user


@pytest.fixture
def mock_user_dict():
    return {"user_id": "user-123", "email": "test@example.com"}


@pytest.fixture
def sample_bookshelf_read():
    now = datetime.now()
    return {
        "user_id": "user-123",
        "book_id": "book-123",
        "shelf_status": "want_to_read",
        "date_added": now,
        "updated_at": now,
        "date_started": None,
        "date_finished": None,
    }


class TestExtractUserId:
    def test_extract_user_id_from_dict_user_id(self):
        assert _extract_user_id({"user_id": "abc"}) == "abc"

    def test_extract_user_id_from_dict_sub(self):
        assert _extract_user_id({"sub": "abc"}) == "abc"

    def test_extract_user_id_from_dict_id(self):
        assert _extract_user_id({"id": "abc"}) == "abc"

    def test_extract_user_id_from_dict_missing(self):
        with pytest.raises(HTTPException) as exc:
            _extract_user_id({"email": "x@test.com"})
        assert exc.value.status_code == 401
        assert "missing user_id" in exc.value.detail.lower()

    def test_extract_user_id_from_object_user_id(self):
        user = MagicMock()
        user.user_id = "abc"
        user.sub = None
        user.id = None
        assert _extract_user_id(user) == "abc"

    def test_extract_user_id_from_object_sub(self):
        user = MagicMock()
        user.user_id = None
        user.sub = "abc-sub"
        user.id = None
        assert _extract_user_id(user) == "abc-sub"

    def test_extract_user_id_from_object_id(self):
        user = MagicMock()
        user.user_id = None
        user.sub = None
        user.id = "abc-id"
        assert _extract_user_id(user) == "abc-id"

    def test_extract_user_id_from_object_missing(self):
        user = MagicMock()
        user.user_id = None
        user.sub = None
        user.id = None
        with pytest.raises(HTTPException) as exc:
            _extract_user_id(user)
        assert exc.value.status_code == 401
        assert "missing user_id" in exc.value.detail.lower()

    def test_extract_user_id_none(self):
        with pytest.raises(HTTPException) as exc:
            _extract_user_id(None)
        assert exc.value.status_code == 401
        assert exc.value.detail == "Not authenticated"


class TestBookshelfRoutes:
    def test_add_book_success(self, client, mock_bookshelf_service, mock_user_obj, sample_bookshelf_read):
        mock_bookshelf_service.add_to_shelf.return_value = sample_bookshelf_read
        app.dependency_overrides[get_current_user] = lambda: mock_user_obj

        with patch("app.routes.bookshelf.get_bookshelf_service", return_value=mock_bookshelf_service):
            response = client.post("/bookshelf/", json={"book_id": "book-123"})

        assert response.status_code == 201
        data = response.json()
        assert data["book_id"] == "book-123"
        assert data["user_id"] == "user-123"
        assert data["shelf_status"] == "want_to_read"
        assert "date_added" in data
        assert "updated_at" in data

        mock_bookshelf_service.add_to_shelf.assert_called_once_with(
            user_id="user-123",
            book_id="book-123",
        )

    def test_add_book_not_found(self, client, mock_bookshelf_service, mock_user_obj):
        mock_bookshelf_service.add_to_shelf.side_effect = ValueError("Book not found")
        app.dependency_overrides[get_current_user] = lambda: mock_user_obj

        with patch("app.routes.bookshelf.get_bookshelf_service", return_value=mock_bookshelf_service):
            response = client.post("/bookshelf/", json={"book_id": "missing-book"})

        assert response.status_code == 404
        assert response.json()["detail"] == "Book not found"

    def test_add_book_duplicate(self, client, mock_bookshelf_service, mock_user_obj):
        mock_bookshelf_service.add_to_shelf.side_effect = ValueError("DUPLICATE")
        app.dependency_overrides[get_current_user] = lambda: mock_user_obj

        with patch("app.routes.bookshelf.get_bookshelf_service", return_value=mock_bookshelf_service):
            response = client.post("/bookshelf/", json={"book_id": "book-123"})

        assert response.status_code == 409
        assert response.json()["detail"] == "Book already exists on shelf"

    def test_add_book_other_value_error(self, client, mock_bookshelf_service, mock_user_obj):
        mock_bookshelf_service.add_to_shelf.side_effect = ValueError("Bad request")
        app.dependency_overrides[get_current_user] = lambda: mock_user_obj

        with patch("app.routes.bookshelf.get_bookshelf_service", return_value=mock_bookshelf_service):
            response = client.post("/bookshelf/", json={"book_id": "book-123"})

        assert response.status_code == 400
        assert response.json()["detail"] == "Bad request"

    def test_add_book_unauthenticated(self, client):
        app.dependency_overrides[get_current_user] = lambda: None

        response = client.post("/bookshelf/", json={"book_id": "book-123"})

        assert response.status_code == 401
        assert response.json()["detail"] == "Not authenticated"

    def test_list_my_shelf_success(self, client, mock_bookshelf_service, mock_user_obj, sample_bookshelf_read):
        mock_bookshelf_service.list_shelf.return_value = [sample_bookshelf_read]
        app.dependency_overrides[get_current_user] = lambda: mock_user_obj

        with patch("app.routes.bookshelf.get_bookshelf_service", return_value=mock_bookshelf_service):
            response = client.get("/bookshelf/")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["book_id"] == "book-123"
        assert data[0]["user_id"] == "user-123"
        assert data[0]["shelf_status"] == "want_to_read"
        assert "date_added" in data[0]
        assert "updated_at" in data[0]

        mock_bookshelf_service.list_shelf.assert_called_once_with(
            user_id="user-123",
            status=None,
            sort="updated_at",
            order="desc",
        )

    def test_list_my_shelf_with_filters(self, client, mock_bookshelf_service, mock_user_obj, sample_bookshelf_read):
        mock_bookshelf_service.list_shelf.return_value = [sample_bookshelf_read]
        app.dependency_overrides[get_current_user] = lambda: mock_user_obj

        with patch("app.routes.bookshelf.get_bookshelf_service", return_value=mock_bookshelf_service):
            response = client.get("/bookshelf/?status=read&sort=date_added&order=asc")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1

        mock_bookshelf_service.list_shelf.assert_called_once_with(
            user_id="user-123",
            status="read",
            sort="date_added",
            order="asc",
        )

    def test_remove_book_success(self, client, mock_bookshelf_service, mock_user_obj):
        mock_bookshelf_service.remove_from_shelf.return_value = None
        app.dependency_overrides[get_current_user] = lambda: mock_user_obj

        with patch("app.routes.bookshelf.get_bookshelf_service", return_value=mock_bookshelf_service):
            response = client.delete("/bookshelf/book-123")

        assert response.status_code == 204
        assert response.content == b""

        mock_bookshelf_service.remove_from_shelf.assert_called_once_with(
            user_id="user-123",
            book_id="book-123",
        )

    def test_remove_book_not_found(self, client, mock_bookshelf_service, mock_user_obj):
        mock_bookshelf_service.remove_from_shelf.side_effect = ValueError("NOT_FOUND")
        app.dependency_overrides[get_current_user] = lambda: mock_user_obj

        with patch("app.routes.bookshelf.get_bookshelf_service", return_value=mock_bookshelf_service):
            response = client.delete("/bookshelf/book-123")

        assert response.status_code == 404
        assert response.json()["detail"] == "Book not found on shelf"

    def test_remove_book_other_error(self, client, mock_bookshelf_service, mock_user_obj):
        mock_bookshelf_service.remove_from_shelf.side_effect = ValueError("Bad request")
        app.dependency_overrides[get_current_user] = lambda: mock_user_obj

        with patch("app.routes.bookshelf.get_bookshelf_service", return_value=mock_bookshelf_service):
            response = client.delete("/bookshelf/book-123")

        assert response.status_code == 400
        assert response.json()["detail"] == "Bad request"

    def test_update_status_success(self, client, mock_bookshelf_service, mock_user_obj, sample_bookshelf_read):
        updated = dict(sample_bookshelf_read)
        updated["shelf_status"] = "read"
        mock_bookshelf_service.update_status.return_value = updated
        app.dependency_overrides[get_current_user] = lambda: mock_user_obj

        with patch("app.routes.bookshelf.get_bookshelf_service", return_value=mock_bookshelf_service):
            response = client.patch("/bookshelf/book-123/status", json={"shelf_status": "read"})

        assert response.status_code == 200
        data = response.json()
        assert data["book_id"] == "book-123"
        assert data["shelf_status"] == "read"
        assert "date_added" in data
        assert "updated_at" in data

        mock_bookshelf_service.update_status.assert_called_once_with(
            user_id="user-123",
            book_id="book-123",
            new_status="read",
        )

    def test_update_status_not_found(self, client, mock_bookshelf_service, mock_user_obj):
        mock_bookshelf_service.update_status.side_effect = ValueError("NOT_FOUND")
        app.dependency_overrides[get_current_user] = lambda: mock_user_obj

        with patch("app.routes.bookshelf.get_bookshelf_service", return_value=mock_bookshelf_service):
            response = client.patch("/bookshelf/book-123/status", json={"shelf_status": "read"})

        assert response.status_code == 404
        assert response.json()["detail"] == "Book not found on shelf"

    def test_update_status_other_error(self, client, mock_bookshelf_service, mock_user_obj):
        mock_bookshelf_service.update_status.side_effect = ValueError("Bad request")
        app.dependency_overrides[get_current_user] = lambda: mock_user_obj

        with patch("app.routes.bookshelf.get_bookshelf_service", return_value=mock_bookshelf_service):
            response = client.patch("/bookshelf/book-123/status", json={"shelf_status": "read"})

        assert response.status_code == 400
        assert response.json()["detail"] == "Bad request"

    def test_timeline_success(self, client, mock_bookshelf_service, mock_user_obj, sample_bookshelf_read):
        mock_bookshelf_service.get_timeline.return_value = [sample_bookshelf_read]
        app.dependency_overrides[get_current_user] = lambda: mock_user_obj

        with patch("app.routes.bookshelf.get_bookshelf_service", return_value=mock_bookshelf_service):
            response = client.get("/bookshelf/timeline")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["book_id"] == "book-123"
        assert "date_added" in data[0]
        assert "updated_at" in data[0]

        mock_bookshelf_service.get_timeline.assert_called_once_with(user_id="user-123")

    def test_stats_success(self, client, mock_bookshelf_service, mock_user_obj):
        mock_bookshelf_service.get_stats.return_value = {
            "want_to_read": 1,
            "currently_reading": 2,
            "read": 3,
            "total": 6,
        }
        app.dependency_overrides[get_current_user] = lambda: mock_user_obj

        with patch("app.routes.bookshelf.get_bookshelf_service", return_value=mock_bookshelf_service):
            response = client.get("/bookshelf/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["want_to_read"] == 1
        assert data["currently_reading"] == 2
        assert data["read"] == 3
        assert data["total"] == 6

        mock_bookshelf_service.get_stats.assert_called_once_with(user_id="user-123")