import unittest
import os
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from fastapi import HTTPException
from datetime import date

from app.main import app
from app.schemas.book import BookCreate, BookRead
from app.services.book_service import BookService


class TestBookRoutes(unittest.TestCase):
    """Integration tests for book CRUD routes."""

    @classmethod
    def setUpClass(cls):
        """Set up class-level patches for Cognito auth and jwks retrieval."""
        from app.services.cognito_service import RoleChecker

        cls.role_patcher = patch.object(RoleChecker, "__call__", return_value={})
        cls.role_patcher.start()

        cls.requests_patcher = patch("app.services.cognito_service.requests.get")
        cls.mock_get = cls.requests_patcher.start()
        cls.mock_get.return_value.status_code = 200
        cls.mock_get.return_value.json.return_value = {"keys": []}

    @classmethod
    def tearDownClass(cls):
        """Clean up class-level patches."""
        cls.role_patcher.stop()
        cls.requests_patcher.stop()

    def setUp(self):
        """Set up test client and mocks."""
        self.client = TestClient(app)

        # Mock book service
        self.mock_book_service = MagicMock(spec=BookService)

        # Sample book data
        self.sample_book = BookRead(
            book_id="test-book-123",
            title="Test Book",
            subtitle="Test Subtitle",
            cover_image_url="https://example.com/cover.jpg",
            abstract="Test abstract",
            CommunitySynopsis="Test synopsis",
            page_count=300,
            published_date=date(2023, 1, 1),
            created_at="2023-01-01T00:00:00Z"
        )

        self.book_create_data = {
            "title": "New Test Book",
            "subtitle": "New Subtitle",
            "cover_image_url": "https://example.com/new-cover.jpg",
            "abstract": "New abstract",
            "page_count": 250,
            "published_date": "2024-01-01"
        }

    @patch('app.routes.books.get_book_service')
    def test_get_books_success(self, mock_get_service):
        """Test GET /books returns list of books."""
        # Arrange
        mock_get_service.return_value = self.mock_book_service
        self.mock_book_service.get_books.return_value = [self.sample_book]

        # Act
        response = self.client.get("/books")

        # Assert
        self.assertEqual(response.status_code, 200)
        books = response.json()
        self.assertEqual(len(books), 1)
        self.assertEqual(books[0]['book_id'], "test-book-123")
        self.assertEqual(books[0]['title'], "Test Book")
        mock_get_service.assert_called_once()
        self.mock_book_service.get_books.assert_called_once()

    @patch('app.routes.books.get_book_service')
    def test_get_book_success(self, mock_get_service):
        """Test GET /books/{book_id} returns specific book."""
        # Arrange
        mock_get_service.return_value = self.mock_book_service
        self.mock_book_service.get_book.return_value = self.sample_book

        # Act
        response = self.client.get("/books/test-book-123")

        # Assert
        self.assertEqual(response.status_code, 200)
        book = response.json()
        self.assertEqual(book['book_id'], "test-book-123")
        self.assertEqual(book['title'], "Test Book")
        mock_get_service.assert_called_once()
        self.mock_book_service.get_book.assert_called_once_with("test-book-123")

    @patch('app.routes.books.get_book_service')
    def test_get_book_not_found(self, mock_get_service):
        """Test GET /books/{book_id} returns 404 for non-existent book."""
        # Arrange
        mock_get_service.return_value = self.mock_book_service
        self.mock_book_service.get_book.return_value = None

        # Act
        response = self.client.get("/books/nonexistent-book")

        # Assert
        self.assertEqual(response.status_code, 404)
        error = response.json()
        self.assertEqual(error['detail'], "Book not found")

    @patch('app.routes.books.get_book_service')
    @patch('app.routes.books.required_admin_role')
    def test_add_book_success(self, mock_admin_role, mock_get_service):
        """Test POST /books creates new book."""
        # Arrange
        mock_get_service.return_value = self.mock_book_service
        mock_admin_role.return_value = None  # Admin role check passes
        self.mock_book_service.add_book.return_value = self.sample_book

        # Act
        response = self.client.post("/books", json=self.book_create_data)

        # Assert
        self.assertEqual(response.status_code, 201)
        book = response.json()
        self.assertEqual(book['book_id'], "test-book-123")
        self.assertEqual(book['title'], "Test Book")
        mock_get_service.assert_called_once()
        self.mock_book_service.add_book.assert_called_once()

    @patch('app.routes.books.get_book_service')
    @patch('app.routes.books.required_admin_role')
    def test_update_book_success(self, mock_admin_role, mock_get_service):
        """Test PUT /books/{book_id} updates book."""
        # Arrange
        mock_get_service.return_value = self.mock_book_service
        mock_admin_role.return_value = None  # Admin role check passes
        self.mock_book_service.update_book.return_value = self.sample_book

        update_data = {"title": "Updated Title", "page_count": 350}

        # Act
        response = self.client.put("/books/test-book-123", json=update_data)

        # Assert
        self.assertEqual(response.status_code, 200)
        book = response.json()
        self.assertEqual(book['book_id'], "test-book-123")
        mock_get_service.assert_called_once()
        self.mock_book_service.update_book.assert_called_once_with("test-book-123", unittest.mock.ANY)

    @patch('app.routes.books.get_book_service')
    @patch('app.routes.books.required_admin_role')
    def test_update_book_not_found(self, mock_admin_role, mock_get_service):
        """Test PUT /books/{book_id} returns 404 for non-existent book."""
        # Arrange
        mock_get_service.return_value = self.mock_book_service
        mock_admin_role.return_value = None  # Admin role check passes
        self.mock_book_service.update_book.return_value = None

        update_data = {"title": "Updated Title"}

        # Act
        response = self.client.put("/books/nonexistent-book", json=update_data)

        # Assert
        self.assertEqual(response.status_code, 404)
        error = response.json()
        self.assertEqual(error['detail'], "Book not found")

    @patch('app.routes.books.get_book_service')
    @patch('app.routes.books.required_admin_role')
    def test_delete_book_success(self, mock_admin_role, mock_get_service):
        """Test DELETE /books/{book_id} deletes book."""
        # Arrange
        mock_get_service.return_value = self.mock_book_service
        mock_admin_role.return_value = None  # Admin role check passes
        self.mock_book_service.delete_book.return_value = True

        # Act
        response = self.client.delete("/books/test-book-123")

        # Assert
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result['message'], "Book deleted successfully")
        mock_get_service.assert_called_once()
        self.mock_book_service.delete_book.assert_called_once_with("test-book-123")

    @patch('app.routes.books.get_book_service')
    @patch('app.routes.books.required_admin_role')
    def test_delete_book_not_found(self, mock_admin_role, mock_get_service):
        """Test DELETE /books/{book_id} returns 404 for non-existent book."""
        # Arrange
        mock_get_service.return_value = self.mock_book_service
        mock_admin_role.return_value = None  # Admin role check passes
        self.mock_book_service.delete_book.return_value = False

        # Act
        response = self.client.delete("/books/nonexistent-book")

        # Assert
        self.assertEqual(response.status_code, 404)
        error = response.json()
        self.assertEqual(error['detail'], "Book not found")


class TestBookRoutesValidation(unittest.TestCase):
    """Tests for book route input validation."""

    def setUp(self):
        """Set up test client."""
        self.client = TestClient(app)

    @patch('app.routes.books.get_book_service')
    @patch('app.routes.books.required_admin_role')
    def test_add_book_invalid_data(self, mock_admin_role, mock_get_service):
        """Test POST /books with invalid data returns validation error."""
        # Arrange
        mock_get_service.return_value = MagicMock()
        mock_admin_role.return_value = None

        # Missing required title field
        invalid_data = {
            "subtitle": "Test Subtitle",
            "page_count": 300
        }

        # Act
        response = self.client.post("/books", json=invalid_data)

        # Assert
        self.assertEqual(response.status_code, 422)  # Validation error

    @patch('app.routes.books.get_book_service')
    @patch('app.routes.books.required_admin_role')
    def test_add_book_empty_title(self, mock_admin_role, mock_get_service):
        """Test POST /books with empty title returns validation error."""
        # Arrange
        mock_get_service.return_value = MagicMock()
        mock_admin_role.return_value = None

        invalid_data = {
            "title": "",  # Empty title
            "page_count": 300
        }

        # Act
        response = self.client.post("/books", json=invalid_data)

        # Assert
        self.assertEqual(response.status_code, 422)  # Validation error


if __name__ == '__main__':
    unittest.main()