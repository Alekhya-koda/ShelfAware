import pytest
from datetime import date, datetime
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

# Patch Cognito-related side effects BEFORE importing app
with patch("app.services.cognito_service.boto3.client") as mock_boto_client, \
     patch("app.services.cognito_service.CognitoService._get_cognito_jwks", return_value={"keys": []}):

    mock_boto_client.return_value = MagicMock()

    from app.main import app

from app.models.book import Book
from app.schemas.book import BookCreate, BookUpdate, BookRead
from app.dependencies.services import get_book_service
from app.dependencies.auth import get_current_user
from app.dependencies.roles import required_admin_role


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_book_service():
    return MagicMock()


@pytest.fixture
def sample_book():
    return Book(
        book_id="test-book-123",
        title="Test Book",
        subtitle="A Test Subtitle",
        cover_image_url="http://example.com/cover.jpg",
        abstract="This is a test book abstract.",
        page_count=300,
        published_date=date(2023, 1, 1)
    )


@pytest.fixture
def sample_book_read():
    return BookRead(
        book_id="test-book-123",
        title="Test Book",
        subtitle="A Test Subtitle",
        cover_image_url="http://example.com/cover.jpg",
        abstract="This is a test book abstract.",
        page_count=300,
        published_date=date(2023, 1, 1),
        created_at=datetime.now()
    )


@pytest.fixture
def sample_book_create():
    return BookCreate(
        title="New Test Book",
        subtitle="New Subtitle",
        cover_image_url="http://example.com/new-cover.jpg",
        abstract="New book abstract.",
        page_count=250,
        published_date=date(2024, 1, 1)
    )


@pytest.fixture
def sample_book_update():
    return BookUpdate(
        title="Updated Test Book",
        page_count=350
    )


@pytest.fixture
def mock_admin_user():
    user = MagicMock()
    user.user_id = "admin-123"
    user.role = "admin"
    return user


@pytest.fixture
def mock_regular_user():
    user = MagicMock()
    user.user_id = "user-123"
    user.role = "user"
    return user


@pytest.fixture(autouse=True)
def clear_dependency_overrides():
    app.dependency_overrides = {}
    yield
    app.dependency_overrides = {}


class TestBookRoutes:
    def test_get_books_success(self, client, mock_book_service, sample_book_read):
        mock_book_service.get_books.return_value = [sample_book_read]
        app.dependency_overrides[get_book_service] = lambda: mock_book_service

        response = client.get("/books")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["book_id"] == "test-book-123"
        assert data[0]["title"] == "Test Book"

    def test_get_book_success(self, client, mock_book_service, sample_book_read):
        mock_book_service.get_book.return_value = sample_book_read
        app.dependency_overrides[get_book_service] = lambda: mock_book_service

        response = client.get("/books/test-book-123")

        assert response.status_code == 200
        data = response.json()
        assert data["book_id"] == "test-book-123"
        assert data["title"] == "Test Book"

    def test_get_book_not_found(self, client, mock_book_service):
        mock_book_service.get_book.return_value = None
        app.dependency_overrides[get_book_service] = lambda: mock_book_service

        response = client.get("/books/nonexistent-book")

        assert response.status_code == 404
        assert "Book not found" in response.json()["detail"]

    def test_add_book_admin_success(
        self, client, mock_book_service, sample_book_create, sample_book_read, mock_admin_user
    ):
        mock_book_service.add_book.return_value = sample_book_read
        app.dependency_overrides[get_book_service] = lambda: mock_book_service
        app.dependency_overrides[get_current_user] = lambda: mock_admin_user
        app.dependency_overrides[required_admin_role] = lambda: mock_admin_user

        response = client.post("/books", json=sample_book_create.model_dump(mode="json"))

        assert response.status_code == 201
        data = response.json()
        assert data["book_id"] == "test-book-123"
        assert data["title"] == "Test Book"

    def test_add_book_regular_user_forbidden(self, client, mock_regular_user, sample_book_create):
        app.dependency_overrides[get_current_user] = lambda: mock_regular_user

        def deny_non_admin():
            raise Exception("This override should not be used directly")

        from fastapi import HTTPException
        app.dependency_overrides[required_admin_role] = lambda: (_ for _ in ()).throw(
            HTTPException(status_code=403, detail="Admin role required")
        )

        response = client.post("/books", json=sample_book_create.model_dump(mode="json"))

        assert response.status_code == 403
        assert "Admin role required" in response.json()["detail"]

    def test_update_book_admin_success(
        self, client, mock_book_service, sample_book_update, sample_book_read, mock_admin_user
    ):
        mock_book_service.update_book.return_value = sample_book_read
        app.dependency_overrides[get_book_service] = lambda: mock_book_service
        app.dependency_overrides[get_current_user] = lambda: mock_admin_user
        app.dependency_overrides[required_admin_role] = lambda: mock_admin_user

        response = client.put(
            "/books/test-book-123",
            json=sample_book_update.model_dump(mode="json", exclude_none=True)
        )

        assert response.status_code == 200
        data = response.json()
        assert data["book_id"] == "test-book-123"

    def test_update_book_not_found(
        self, client, mock_book_service, sample_book_update, mock_admin_user
    ):
        mock_book_service.update_book.return_value = None
        app.dependency_overrides[get_book_service] = lambda: mock_book_service
        app.dependency_overrides[get_current_user] = lambda: mock_admin_user
        app.dependency_overrides[required_admin_role] = lambda: mock_admin_user

        response = client.put(
            "/books/nonexistent-book",
            json=sample_book_update.model_dump(mode="json", exclude_none=True)
        )

        assert response.status_code == 404
        assert "Book not found" in response.json()["detail"]

    def test_update_book_regular_user_forbidden(self, client, mock_regular_user, sample_book_update):
        from fastapi import HTTPException

        app.dependency_overrides[get_current_user] = lambda: mock_regular_user
        app.dependency_overrides[required_admin_role] = lambda: (_ for _ in ()).throw(
            HTTPException(status_code=403, detail="Admin role required")
        )

        response = client.put(
            "/books/test-book-123",
            json=sample_book_update.model_dump(mode="json", exclude_none=True)
        )

        assert response.status_code == 403
        assert "Admin role required" in response.json()["detail"]

    def test_delete_book_admin_success(self, client, mock_book_service, mock_admin_user):
        mock_book_service.delete_book.return_value = True
        app.dependency_overrides[get_book_service] = lambda: mock_book_service
        app.dependency_overrides[get_current_user] = lambda: mock_admin_user
        app.dependency_overrides[required_admin_role] = lambda: mock_admin_user

        response = client.delete("/books/test-book-123")

        assert response.status_code == 200

    def test_delete_book_not_found(self, client, mock_book_service, mock_admin_user):
        mock_book_service.delete_book.return_value = False
        app.dependency_overrides[get_book_service] = lambda: mock_book_service
        app.dependency_overrides[get_current_user] = lambda: mock_admin_user
        app.dependency_overrides[required_admin_role] = lambda: mock_admin_user

        response = client.delete("/books/nonexistent-book")

        assert response.status_code == 404
        assert "Book not found" in response.json()["detail"]

    def test_delete_book_regular_user_forbidden(self, client, mock_regular_user):
        from fastapi import HTTPException

        app.dependency_overrides[get_current_user] = lambda: mock_regular_user
        app.dependency_overrides[required_admin_role] = lambda: (_ for _ in ()).throw(
            HTTPException(status_code=403, detail="Admin role required")
        )

        response = client.delete("/books/test-book-123")

        assert response.status_code == 403
        assert "Admin role required" in response.json()["detail"]

    def test_add_book_validation_error(self, client, mock_admin_user):
        app.dependency_overrides[get_current_user] = lambda: mock_admin_user
        app.dependency_overrides[required_admin_role] = lambda: mock_admin_user

        invalid_data = {"subtitle": "Test Subtitle"}

        response = client.post("/books", json=invalid_data)

        assert response.status_code == 422
        assert "title" in str(response.json())

    def test_update_book_validation_error(self, client, mock_book_service, mock_admin_user):
        app.dependency_overrides[get_book_service] = lambda: mock_book_service
        app.dependency_overrides[get_current_user] = lambda: mock_admin_user
        app.dependency_overrides[required_admin_role] = lambda: mock_admin_user

        invalid_data = {"page_count": -1}

        response = client.put("/books/test-book-123", json=invalid_data)

        assert response.status_code == 422