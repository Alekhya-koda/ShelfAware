"""
Shared pytest fixtures and configuration for review tests.
This file is automatically discovered by pytest.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock
from sqlalchemy.orm import Session

from app.services.review_service import ReviewService
from app.models.review import Review
from app.models.user import User
from app.models.book import Book
from app.models.mood import Mood


@pytest.fixture
def mock_db():
    """Mock SQLAlchemy database session."""
    return Mock(spec=Session)


@pytest.fixture
def review_service(mock_db):
    """Create ReviewService with mocked DB."""
    return ReviewService(db=mock_db)


@pytest.fixture
def sample_user():
    """Sample user data."""
    return {
        "user_id": "user-123",
        "cognito_sub": "cognito-sub-123",
        "id": "user-123",
    }


@pytest.fixture
def sample_book():
    """Sample book data."""
    return {
        "book_id": "book-456",
        "title": "Sample Book",
    }


@pytest.fixture
def sample_review():
    """Sample review object."""
    return Review(
        review_id="review-789",
        user_id="user-123",
        book_id="book-456",
        rating=5,
        title="Great Book!",
        body="This book was amazing!",
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )


@pytest.fixture
def sample_review_low_rating():
    """Sample review with low rating."""
    return Review(
        review_id="review-790",
        user_id="user-124",
        book_id="book-456",
        rating=2,
        title="Not Great",
        body="Disappointed with this book.",
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )


@pytest.fixture
def sample_reviews_list():
    """Sample list of reviews."""
    return [
        Review(
            review_id=f"review-{i}",
            user_id=f"user-{i}",
            book_id="book-456",
            rating=5 - (i % 5),
            title=f"Review {i}",
            body=f"This is review number {i}",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        for i in range(5)
    ]