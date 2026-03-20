import pytest
from unittest.mock import Mock, MagicMock, patch
from sqlalchemy.orm import Session
from app.services.book_service import BookService
from app.models.book import Book
from app.schemas.book import BookCreate, BookUpdate
from datetime import date


@pytest.fixture
def mock_db():
    """Mock SQLAlchemy session."""
    return MagicMock(spec=Session)


@pytest.fixture
def book_service(mock_db):
    """BookService instance with mocked database."""
    return BookService(mock_db)


@pytest.fixture
def sample_book():
    """Sample book instance for testing."""
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
def sample_book_create():
    """Sample BookCreate schema for testing."""
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
    """Sample BookUpdate schema for testing."""
    return BookUpdate(
        title="Updated Test Book",
        page_count=350
    )


class TestBookService:
    """Unit tests for BookService CRUD operations."""

    def test_get_books_no_limit(self, book_service, mock_db, sample_book):
        """Test get_books without limit."""
        mock_db.query.return_value.all.return_value = [sample_book]

        result = book_service.get_books()

        mock_db.query.assert_called_once_with(Book)
        mock_db.query.return_value.all.assert_called_once()
        assert result == [sample_book]

    def test_get_books_with_limit(self, book_service, mock_db, sample_book):
        """Test get_books with limit."""
        mock_db.query.return_value.limit.return_value.all.return_value = [sample_book]

        result = book_service.get_books(limit=10)

        mock_db.query.assert_called_once_with(Book)
        mock_db.query.return_value.limit.assert_called_once_with(10)
        mock_db.query.return_value.limit.return_value.all.assert_called_once()
        assert result == [sample_book]

    def test_get_book_found(self, book_service, mock_db, sample_book):
        """Test get_book when book exists."""
        mock_db.query.return_value.filter.return_value.first.return_value = sample_book

        result = book_service.get_book("test-book-123")

        mock_db.query.assert_called_once_with(Book)
        mock_db.query.return_value.filter.assert_called_once()
        mock_db.query.return_value.filter.return_value.first.assert_called_once()
        assert result == sample_book

    def test_get_book_not_found(self, book_service, mock_db):
        """Test get_book when book does not exist."""
        mock_db.query.return_value.filter.return_value.first.return_value = None

        result = book_service.get_book("nonexistent-book")

        assert result is None

    def test_add_book(self, book_service, mock_db, sample_book_create):
        """Test add_book creates and saves a new book."""
        # Mock the new book instance that would be created
        new_book = Book(**sample_book_create.model_dump())
        new_book.book_id = "generated-id-123"
        mock_db.add.return_value = None
        mock_db.commit.return_value = None
        mock_db.refresh.return_value = None

        # Mock the constructor to return our prepared book
        with patch('app.services.book_service.Book', return_value=new_book):
            result = book_service.add_book(sample_book_create)

        mock_db.add.assert_called_once_with(new_book)
        mock_db.commit.assert_called_once()
        mock_db.refresh.assert_called_once_with(new_book)
        assert result == new_book

    def test_update_book_found(self, book_service, mock_db, sample_book, sample_book_update):
        """Test update_book when book exists."""
        # Setup mock to return the book
        mock_db.query.return_value.filter.return_value.first.return_value = sample_book

        result = book_service.update_book("test-book-123", sample_book_update)

        # Verify the book attributes were updated
        assert sample_book.title == "Updated Test Book"
        assert sample_book.page_count == 350
        # Other attributes should remain unchanged
        assert sample_book.subtitle == "A Test Subtitle"

        mock_db.commit.assert_called_once()
        mock_db.refresh.assert_called_once_with(sample_book)
        assert result == sample_book

    def test_update_book_not_found(self, book_service, mock_db, sample_book_update):
        """Test update_book when book does not exist."""
        mock_db.query.return_value.filter.return_value.first.return_value = None

        result = book_service.update_book("nonexistent-book", sample_book_update)

        assert result is None
        mock_db.commit.assert_not_called()
        mock_db.refresh.assert_not_called()

    def test_delete_book_found(self, book_service, mock_db, sample_book):
        """Test delete_book when book exists."""
        mock_db.query.return_value.filter.return_value.first.return_value = sample_book

        result = book_service.delete_book("test-book-123")

        mock_db.delete.assert_called_once_with(sample_book)
        mock_db.commit.assert_called_once()
        assert result is True

    def test_delete_book_not_found(self, book_service, mock_db):
        """Test delete_book when book does not exist."""
        mock_db.query.return_value.filter.return_value.first.return_value = None

        result = book_service.delete_book("nonexistent-book")

        mock_db.delete.assert_not_called()
        mock_db.commit.assert_not_called()
        assert result is False