# Test to create, update, delete, and retrieve reviews.


import pytest
from fastapi import status, HTTPException
from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError
from app.services.review_service import ReviewService
from app.schemas.review import ReviewCreate, ReviewUpdate


class TestAddReview:
    """Tests for ReviewService.add_review()"""

    def test_add_review_success(self, review_service, mock_db):
        """Test successfully adding a review with valid data."""
        # ARRANGE
        mock_db.scalar.side_effect = ["book-456", "user-123"]
        mock_db.commit = lambda: None
        mock_db.refresh = lambda x: None
        review_data = ReviewCreate(rating=5, comment="Excellent book!")
        
        # ACT
        result = review_service.add_review(
            book_id="book-456",
            user_id="user-123",
            review_data=review_data
        )
        
        # ASSERT
        assert result is not None
        assert result.rating == 5
        mock_db.add.assert_called()

    def test_review_create_schema_rating_too_low(self):
        """Test ReviewCreate schema rejects rating < 1."""
        # ARRANGE & ACT & ASSERT
        # Pydantic validation happens at schema creation time
        with pytest.raises(ValidationError) as exc_info:
            ReviewCreate(rating=0, comment="Bad")
        
        # ASSERT: Verify validation error
        errors = exc_info.value.errors()
        assert any(error['loc'][0] == 'rating' for error in errors)

    def test_review_create_schema_rating_too_high(self):
        """Test ReviewCreate schema rejects rating > 5."""
        # ARRANGE & ACT & ASSERT
        with pytest.raises(ValidationError) as exc_info:
            ReviewCreate(rating=6, comment="Too good")
        
        # ASSERT: Verify validation error
        errors = exc_info.value.errors()
        assert any("less than or equal to 5" in str(e) for e in errors)

    def test_review_create_schema_negative_rating(self):
        """Test ReviewCreate schema rejects negative rating."""
        # ARRANGE & ACT & ASSERT
        with pytest.raises(ValidationError):
            ReviewCreate(rating=-1, comment="Invalid")

    def test_add_review_minimum_valid_rating(self, review_service, mock_db):
        """Test adding review with minimum valid rating (1)."""
        # ARRANGE
        mock_db.scalar.side_effect = ["book-456", "user-123"]
        mock_db.commit = lambda: None
        mock_db.refresh = lambda x: None
        review_data = ReviewCreate(rating=1, comment="Terrible")
        
        # ACT
        result = review_service.add_review(
            book_id="book-456",
            user_id="user-123",
            review_data=review_data
        )
        
        # ASSERT
        assert result is not None
        assert result.rating == 1

    def test_add_review_maximum_valid_rating(self, review_service, mock_db):
        """Test adding review with maximum valid rating (5)."""
        # ARRANGE
        mock_db.scalar.side_effect = ["book-456", "user-123"]
        mock_db.commit = lambda: None
        mock_db.refresh = lambda x: None
        review_data = ReviewCreate(rating=5, comment="Perfect!")
        
        # ACT
        result = review_service.add_review(
            book_id="book-456",
            user_id="user-123",
            review_data=review_data
        )
        
        # ASSERT
        assert result is not None
        assert result.rating == 5

    def test_add_review_book_not_found(self, review_service, mock_db):
        """Test adding review when book doesn't exist."""
        # ARRANGE
        mock_db.scalar.return_value = None  # Book not found
        review_data = ReviewCreate(rating=5, comment="Test")
        
        # ACT & ASSERT
        with pytest.raises(HTTPException) as exc_info:
            review_service.add_review(
                book_id="nonexistent-book",
                user_id="user-123",
                review_data=review_data
            )
        
        # ASSERT
        assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND
        assert "Book not found" in exc_info.value.detail

    def test_add_review_user_not_found(self, review_service, mock_db):
        """Test adding review when user doesn't exist."""
        # ARRANGE
        mock_db.scalar.side_effect = ["book-456", None]  # Book exists, user not found
        review_data = ReviewCreate(rating=5, comment="Test")
        
        # ACT & ASSERT
        with pytest.raises(HTTPException) as exc_info:
            review_service.add_review(
                book_id="book-456",
                user_id="nonexistent-user",
                review_data=review_data
            )
        
        # ASSERT
        assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND
        assert "User not found" in exc_info.value.detail

    def test_add_review_duplicate_constraint_violation(self, review_service, mock_db):
        """Test adding duplicate review for same user-book combination."""
        # ARRANGE
        mock_db.scalar.side_effect = ["book-456", "user-123"]
        mock_db.commit.side_effect = IntegrityError("Unique constraint", None, None)
        mock_db.rollback = lambda: None
        review_data = ReviewCreate(rating=5, comment="Test")
        
        # ACT & ASSERT
        with pytest.raises(HTTPException) as exc_info:
            review_service.add_review(
                book_id="book-456",
                user_id="user-123",
                review_data=review_data
            )
        
        # ASSERT
        assert exc_info.value.status_code == status.HTTP_409_CONFLICT
        assert "already reviewed" in exc_info.value.detail.lower()

    def test_add_review_with_mood(self, review_service, mock_db):
        """Test adding review with mood data."""
        # ARRANGE
        mock_db.scalar.side_effect = ["book-456", "user-123"]
        mock_db.commit = lambda: None
        mock_db.refresh = lambda x: None
        review_data = ReviewCreate(rating=5, comment="Great!", mood="happy")
        
        # ACT
        review_service.add_review(
            book_id="book-456",
            user_id="user-123",
            review_data=review_data
        )
        
        # ASSERT: Verify mood entry was added
        calls = [call[0][0] for call in mock_db.add.call_args_list]
        from app.models.mood import Mood
        mood_added = any(isinstance(call, Mood) for call in calls)
        assert mood_added


class TestUpdateReview:
    """Tests for ReviewService.update_review()"""

    def test_update_review_success(self, review_service, mock_db, sample_review):
        """Test successfully updating a review."""
        # ARRANGE
        mock_db.scalar.return_value = "user-123"  # User exists
        mock_db.get.return_value = sample_review  # Review found
        mock_db.commit = lambda: None
        mock_db.refresh = lambda x: None
        update_data = ReviewUpdate(rating=4, comment="Updated comment")
        
        # ACT
        result = review_service.update_review(
            review_id="review-789",
            acting_user_id="user-123",
            review_data=update_data
        )
        
        # ASSERT
        assert result is not None
        assert result.rating == 4

    def test_update_review_rating_only(self, review_service, mock_db, sample_review):
        """Test updating only the rating."""
        # ARRANGE
        mock_db.scalar.return_value = "user-123"
        mock_db.get.return_value = sample_review
        mock_db.commit = lambda: None
        mock_db.refresh = lambda x: None
        update_data = ReviewUpdate(rating=3)
        
        # ACT
        result = review_service.update_review(
            review_id="review-789",
            acting_user_id="user-123",
            review_data=update_data
        )
        
        # ASSERT
        assert result is not None

    def test_update_review_comment_only(self, review_service, mock_db, sample_review):
        """Test updating only the comment."""
        # ARRANGE
        mock_db.scalar.return_value = "user-123"
        mock_db.get.return_value = sample_review
        mock_db.commit = lambda: None
        mock_db.refresh = lambda x: None
        update_data = ReviewUpdate(comment="Updated text")
        
        # ACT
        result = review_service.update_review(
            review_id="review-789",
            acting_user_id="user-123",
            review_data=update_data
        )
        
        # ASSERT
        assert result is not None

    def test_review_update_schema_invalid_rating(self):
        """Test ReviewUpdate schema rejects invalid rating."""
        # ARRANGE & ACT & ASSERT
        with pytest.raises(ValidationError):
            ReviewUpdate(rating=6)

    def test_update_review_not_authorized(self, review_service, mock_db, sample_review):
        """Test updating review by non-owner user."""
        # ARRANGE
        mock_db.scalar.return_value = "different-user"
        mock_db.get.return_value = sample_review
        update_data = ReviewUpdate(rating=1)
        
        # ACT & ASSERT
        with pytest.raises(HTTPException) as exc_info:
            review_service.update_review(
                review_id="review-789",
                acting_user_id="different-user",
                review_data=update_data
            )
        
        # ASSERT
        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
        assert "Not authorized" in exc_info.value.detail

    def test_update_review_not_found(self, review_service, mock_db):
        """Test updating non-existent review."""
        # ARRANGE
        mock_db.scalar.return_value = "user-123"
        mock_db.get.return_value = None  # Review not found
        update_data = ReviewUpdate(rating=5)
        
        # ACT & ASSERT
        with pytest.raises(HTTPException) as exc_info:
            review_service.update_review(
                review_id="nonexistent-review",
                acting_user_id="user-123",
                review_data=update_data
            )
        
        # ASSERT
        assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND


class TestDeleteReview:
    """Tests for ReviewService.delete_review()"""

    def test_delete_review_success(self, review_service, mock_db, sample_review):
        """Test successfully deleting a review."""
        # ARRANGE
        mock_db.scalar.return_value = "user-123"
        mock_db.get.return_value = sample_review
        mock_db.delete = lambda x: None
        mock_db.commit = lambda: None
        
        # ACT
        review_service.delete_review(
            review_id="review-789",
            acting_user_id="user-123"
        )
        
        # ASSERT (No exception = success)
        assert True

    def test_delete_review_not_authorized(self, review_service, mock_db, sample_review):
        """Test deleting review by non-owner."""
        # ARRANGE
        mock_db.scalar.return_value = "different-user"
        mock_db.get.return_value = sample_review
        
        # ACT & ASSERT
        with pytest.raises(HTTPException) as exc_info:
            review_service.delete_review(
                review_id="review-789",
                acting_user_id="different-user"
            )
        
        # ASSERT
        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN

    def test_delete_review_not_found(self, review_service, mock_db):
        """Test deleting non-existent review."""
        # ARRANGE
        mock_db.scalar.return_value = "user-123"
        mock_db.get.return_value = None
        
        # ACT & ASSERT
        with pytest.raises(HTTPException) as exc_info:
            review_service.delete_review(
                review_id="nonexistent",
                acting_user_id="user-123"
            )
        
        # ASSERT
        assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND


class TestGetReviewsByBookId:
    """Tests for ReviewService.get_reviews_by_book_id()"""

    def test_get_reviews_by_book_id_success(self, review_service, mock_db, sample_review):
        """Test retrieving reviews for a book."""
        # ARRANGE
        mock_db.scalar.return_value = "book-456"  # Book exists
        mock_db.scalars.return_value.all.return_value = [sample_review]
        
        # ACT
        result = review_service.get_reviews_by_book_id(
            book_id="book-456",
            limit=20,
            offset=0,
            newest_first=True
        )
        
        # ASSERT
        assert len(result) == 1
        assert result[0].review_id == "review-789"

    def test_get_reviews_by_book_id_empty_result(self, review_service, mock_db):
        """Test retrieving reviews when none exist."""
        # ARRANGE
        mock_db.scalar.return_value = "book-456"
        mock_db.scalars.return_value.all.return_value = []
        
        # ACT
        result = review_service.get_reviews_by_book_id(book_id="book-456")
        
        # ASSERT
        assert len(result) == 0

    def test_get_reviews_by_book_id_with_pagination(self, review_service, mock_db, sample_reviews_list):
        """Test pagination in get_reviews_by_book_id."""
        # ARRANGE
        mock_db.scalar.return_value = "book-456"
        mock_db.scalars.return_value.all.return_value = sample_reviews_list[:2]
        
        # ACT
        result = review_service.get_reviews_by_book_id(
            book_id="book-456",
            limit=2,
            offset=0,
            newest_first=True
        )
        
        # ASSERT
        assert len(result) == 2

    def test_get_reviews_by_book_id_not_found(self, review_service, mock_db):
        """Test retrieving reviews when book doesn't exist."""
        # ARRANGE
        mock_db.scalar.return_value = None
        
        # ACT & ASSERT
        with pytest.raises(HTTPException) as exc_info:
            review_service.get_reviews_by_book_id(book_id="nonexistent")
        
        # ASSERT
        assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND


class TestGetAverageRating:
    """Tests for ReviewService.get_average_rating()"""

    def test_get_average_rating_success(self, review_service, mock_db):
        """Test getting average rating for a book."""
        # ARRANGE
        mock_db.scalar.side_effect = ["book-456", 4.5]
        
        # ACT
        result = review_service.get_average_rating(book_id="book-456")
        
        # ASSERT
        assert result == 4.5

    def test_get_average_rating_perfect_score(self, review_service, mock_db):
        """Test average rating when all reviews are 5 stars."""
        # ARRANGE
        mock_db.scalar.side_effect = ["book-456", 5.0]
        
        # ACT
        result = review_service.get_average_rating(book_id="book-456")
        
        # ASSERT
        assert result == 5.0

    def test_get_average_rating_no_reviews(self, review_service, mock_db):
        """Test average rating when no reviews exist."""
        # ARRANGE
        mock_db.scalar.side_effect = ["book-456", None]
        
        # ACT
        result = review_service.get_average_rating(book_id="book-456")
        
        # ASSERT
        assert result is None

    def test_get_average_rating_book_not_found(self, review_service, mock_db):
        """Test average rating when book doesn't exist."""
        # ARRANGE
        mock_db.scalar.return_value = None
        
        # ACT & ASSERT
        with pytest.raises(HTTPException) as exc_info:
            review_service.get_average_rating(book_id="nonexistent")
        
        # ASSERT
        assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND