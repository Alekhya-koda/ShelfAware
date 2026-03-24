import os
from datetime import datetime
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from app.dependencies.auth import get_current_user
from app.main import app
from app.services.review_service import ReviewService


@pytest.fixture(autouse=True)
def mock_env_vars():
    """Mock environment variables."""
    with patch.dict(
        os.environ,
        {
            "DATABASE_URL": "sqlite:///test.db",
        },
    ):
        yield


@pytest.fixture
def mock_review_service():
    """Mock ReviewService for route testing."""
    with patch("app.routes.review.ReviewService") as MockService:
        mock_instance = Mock(spec=ReviewService)
        MockService.return_value = mock_instance
        yield MockService


@pytest.fixture
def client(mock_review_service):
    """Test client with mocked auth and review service."""

    async def mock_get_current_user():
        return {
            "user_id": "test-user-123",
            "email": "test@example.com",
            "role": "user",
        }

    app.dependency_overrides[get_current_user] = mock_get_current_user
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ============================================================================
# TESTS: POST /reviews/books/{book_id}
# ============================================================================


class TestCreateReviewEndpoint:
    """Tests for POST /reviews/books/{book_id}."""

    def test_create_review_success(self, client, mock_review_service):
        """Test successfully creating a review."""
        # Arrange
        mock_instance = mock_review_service.return_value
        mock_instance.add_review.return_value = {
            "review_id": "review-789",
            "book_id": "book-456",
            "user_id": "user-123",
            "rating": 5,
            "comment": "Great book!",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }

        payload = {"rating": 5, "comment": "Great book!"}

        # Act
        response = client.post("/reviews/books/book-456", json=payload)

        # Assert
        assert response.status_code == 201
        data = response.json()
        assert data["rating"] == 5
        assert data["comment"] == "Great book!"
        mock_instance.add_review.assert_called_once()

    def test_create_review_invalid_rating_zero(self, client, mock_review_service):
        """Test creating review with invalid rating."""
        # Arrange
        payload = {"rating": 0, "comment": "Bad"}

        # Act
        response = client.post("/reviews/books/book-456", json=payload)

        # Assert
        assert response.status_code == 422  # Validation error

    def test_create_review_missing_rating(self, client, mock_review_service):
        """Test creating review without required rating."""
        # Arrange
        payload = {"comment": "No rating"}

        # Act
        response = client.post("/reviews/books/book-456", json=payload)

        # Assert
        assert response.status_code == 422

    def test_create_review_book_not_found(self, client, mock_review_service):
        """Test creating review when book doesn't exist."""
        from fastapi import HTTPException

        # Arrange
        mock_instance = mock_review_service.return_value
        mock_instance.add_review.side_effect = HTTPException(
            status_code=404,
            detail="Book not found",
        )

        payload = {"rating": 5, "comment": "Test"}

        # Act
        response = client.post("/reviews/books/nonexistent", json=payload)

        # Assert
        assert response.status_code == 404

# Tests for auth fallback to cognito sub when user_id is missing, and related error cases
def test_create_review_uses_sub_when_no_user_id(client, mock_review_service, db):
    from app.models.user import User
    from app.dependencies.auth import get_current_user

    # Create user in DB with a cognito_sub
    user = User(cognito_sub="cognito-sub-xyz", email="sub@test.com")
    db.add(user)
    db.commit()

    async def mock_user_with_sub():
        return {"sub": "cognito-sub-xyz"}  # no user_id, only sub

    app.dependency_overrides[get_current_user] = mock_user_with_sub

    mock_review_service.return_value.add_review.return_value = Mock()
    client.post("/reviews/books/book-456", json={"rating": 4, "comment": "test"})

# Test for auth failures related to missing/invalid sub when user_id is not provided
def test_create_review_raises_401_when_sub_missing(client, mock_review_service):
    from app.dependencies.auth import get_current_user

    async def mock_empty_user():
        return {}  # no user_id, no sub

    app.dependency_overrides[get_current_user] = mock_empty_user

    response = client.post("/reviews/books/book-456", json={"rating": 4, "comment": "test"})
    assert response.status_code == 401
    assert "missing user_id/sub" in response.json()["detail"]

# Test for auth failures when sub is provided but not found in DB
def test_create_review_raises_401_when_sub_not_in_db(client, mock_review_service):
    from app.dependencies.auth import get_current_user

    async def mock_unknown_sub():
        return {"sub": "nonexistent-sub"}

    app.dependency_overrides[get_current_user] = mock_unknown_sub

    response = client.post("/reviews/books/book-456", json={"rating": 4, "comment": "test"})
    assert response.status_code == 401
    assert "not found in database" in response.json()["detail"]

# Test fetching reviews return 200 with empty list when no reviews exist for the book
def test_get_reviews_for_book_success(client, mock_review_service):
    mock_review_service.return_value.get_reviews_by_book_id.return_value = []
    response = client.get("/reviews/book/book-456")
    assert response.status_code == 200
    assert response.json() == []

#Test that pagination parameters are correctly passed to the service layer when fetching reviews for a book
def test_get_reviews_for_book_with_pagination(client, mock_review_service):
    mock_review_service.return_value.get_reviews_by_book_id.return_value = []
    response = client.get("/reviews/book/book-456?limit=5&offset=10&newest_first=false")
    assert response.status_code == 200
    mock_review_service.return_value.get_reviews_by_book_id.assert_called_once_with(
        book_id="book-456", limit=5, offset=10, newest_first=False
    )

#Test fetching a specific review return 200 when the review exists
def test_get_review_success(client, mock_review_service):
    from unittest.mock import Mock
    mock_review = Mock()
    mock_review_service.return_value._get_review_or_404.return_value = mock_review
    response = client.get("/reviews/review-789")
    assert response.status_code == 200

#Test fetching a specific review return 404 when the review does not exist
def test_get_review_not_found(client, mock_review_service):
    from fastapi import HTTPException
    mock_review_service.return_value._get_review_or_404.side_effect = HTTPException(
        status_code=404, detail="Review not found"
    )
    response = client.get("/reviews/nonexistent")
    assert response.status_code == 404

# Test updating a review return 200 when successful
def test_update_review_success(client, mock_review_service):
    from unittest.mock import Mock
    mock_review_service.return_value.update_review.return_value = Mock()
    payload = {"rating": 4, "comment": "Updated review"}
    response = client.put("/reviews/review-789", json=payload)
    assert response.status_code == 200

# Test updating a review return 404 when the review does not exist
def test_update_review_not_found(client, mock_review_service):
    from fastapi import HTTPException
    mock_review_service.return_value.update_review.side_effect = HTTPException(
        status_code=404, detail="Review not found"
    )
    response = client.put("/reviews/nonexistent", json={"rating": 3, "comment": "test"})
    assert response.status_code == 404

#Test updating a review return 403 when the user is not the owner of the review
def test_update_review_forbidden(client, mock_review_service):
    from fastapi import HTTPException
    mock_review_service.return_value.update_review.side_effect = HTTPException(
        status_code=403, detail="Not authorized"
    )
    response = client.put("/reviews/review-789", json={"rating": 3, "comment": "test"})
    assert response.status_code == 403

#Test deleting a review return 204 when successful
def test_delete_review_success(client, mock_review_service):
    mock_review_service.return_value.delete_review.return_value = None
    response = client.delete("/reviews/review-789")
    assert response.status_code == 204

#Test deleting a review return 404 when the review does not exist
def test_delete_review_not_found(client, mock_review_service):
    from fastapi import HTTPException
    mock_review_service.return_value.delete_review.side_effect = HTTPException(
        status_code=404, detail="Review not found"
    )
    response = client.delete("/reviews/nonexistent")
    assert response.status_code == 404

#Test deleting a review return 403 when the user is not the owner of the review
def test_delete_review_forbidden(client, mock_review_service):
    from fastapi import HTTPException
    mock_review_service.return_value.delete_review.side_effect = HTTPException(
        status_code=403, detail="Not authorized"
    )
    response = client.delete("/reviews/review-789")
    assert response.status_code == 403