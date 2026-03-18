import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, call
import os
import uuid
from typing import Optional

from app.main import app
from app.services.chroma_service import ChromaService
from app.models.book import Book
from app.services.book_service import BookService
from app.dependencies.auth import get_current_user

# Mock environment variables - crucial for ChromaService initialization
@pytest.fixture(autouse=True)
def mock_env_vars():
    with patch.dict(os.environ, {
        "OPENAI_API_KEY": "test_openai_key",
        "OPENAI_EMBEDDING_MODEL": "test_embedding_model",
        "OPENAI_LLM_MODEL": "test_llm_model",
        "LLM_PROVIDER": "OPENAI"
    }):
        yield

# Mock ChromaDB components
@pytest.fixture
def mock_chroma_client():
    with patch("chromadb.PersistentClient") as MockPersistentClient:
        mock_client = Mock()
        MockPersistentClient.return_value = mock_client
        yield mock_client

@pytest.fixture
def mock_chroma_collection():
    mock_collection = Mock()
    mock_collection.name = "books"
    return mock_collection

# Fixture to provide a correctly mocked ChromaService instance
@pytest.fixture
def chroma_service_for_sync(mock_chroma_client, mock_chroma_collection):
    # Bypass the original __init__ to avoid complex setup with LLM clients
    with patch.object(ChromaService, '__init__', lambda s, llm_provider_override=None: None):
        service = ChromaService()
        # Manually attach the mocks needed for the sync tests
        service.client = mock_chroma_client
        service.collection = mock_chroma_collection
        yield service

# Mock database session dependency
@pytest.fixture
def mock_db_session():
    mock_session = Mock()
    # Mock the get_db dependency to yield our mock session
    with patch('app.services.chroma_service.get_db', return_value=iter([mock_session])):
        yield mock_session

# Mock BookService class
@pytest.fixture
def mock_book_service():
    return Mock(spec=BookService)

# Autouse fixture to patch BookService instantiation in all tests
@pytest.fixture(autouse=True)
def patch_book_service_instantiation(mock_book_service):
    # When `BookService(db)` is called inside `sync_books`, it will return our mock_book_service instance
    with patch('app.services.chroma_service.BookService') as MockBookService:
        MockBookService.return_value = mock_book_service
        yield

# Helper function to create a mock Book object
def create_mock_book(book_id: str, title: str, abstract: Optional[str]):
    mock_book = Mock(spec=Book)
    mock_book.book_id = book_id
    mock_book.title = title
    mock_book.abstract = abstract
    return mock_book

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.db.database import Base, get_db


# --- New fixtures for Route Testing ---

@pytest.fixture(scope="function")
def test_db_session():
    """
    Fixture to create a new in-memory SQLite database session for each test function.
    It overrides the `get_db` dependency to ensure test isolation.
    """
    TEST_DATABASE_URL = "sqlite:///:memory:"
    engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    # Replace the app's dependency with the test version
    app.dependency_overrides[get_db] = override_get_db
    
    yield

    # Clean up by dropping all tables and clearing the override
    Base.metadata.drop_all(bind=engine)
    app.dependency_overrides.clear()


@pytest.fixture
def client(test_db_session):
    """
    Fixture to create a TestClient that uses the isolated test database.
    It also mocks the user authentication.
    """
    # Mock authentication to allow access to the endpoint
    async def mock_get_current_user():
        return {"id": "test_user", "email": "test@example.com", "role": "admin"}
    
    app.dependency_overrides[get_current_user] = mock_get_current_user
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()

@pytest.fixture
def mock_chroma_service_in_route():
    # Patch ChromaService where it is imported in the routes module
    with patch('app.routes.chroma.ChromaService') as MockChromaService:
        mock_instance = Mock(spec=ChromaService)
        mock_instance.llm_provider = "OPENAI" # Default for mock
        MockChromaService.return_value = mock_instance
        yield MockChromaService # Return the CLASS mock to check constructor calls

# --- Tests for sync_books ---

def test_sync_books_no_books_in_db_or_chroma(
    chroma_service_for_sync, mock_db_session, mock_book_service, mock_chroma_collection
):
    # Arrange
    mock_book_service.get_books.return_value = []
    mock_chroma_collection.get.return_value = {"ids": []}

    # Act
    result = chroma_service_for_sync.sync_books()

    # Assert
    assert result == {"upserted": 0, "deleted": 0}
    mock_book_service.get_books.assert_called_once_with(limit=None)
    mock_chroma_collection.get.assert_called_once()
    mock_chroma_collection.upsert.assert_not_called()
    mock_chroma_collection.delete.assert_not_called()

def test_sync_books_add_new_books_to_chroma(
    chroma_service_for_sync, mock_db_session, mock_book_service, mock_chroma_collection
):
    # Arrange
    book_id_1 = str(uuid.uuid4())
    book_id_2 = str(uuid.uuid4())
    db_books = [
        create_mock_book(book_id_1, "Title 1", "Abstract 1"),
        create_mock_book(book_id_2, "Title 2", None),
    ]
    mock_book_service.get_books.return_value = db_books
    mock_chroma_collection.get.return_value = {"ids": []}

    # Act
    result = chroma_service_for_sync.sync_books()

    # Assert
    assert result == {"upserted": 2, "deleted": 0}
    mock_book_service.get_books.assert_called_once_with(limit=None)
    mock_chroma_collection.get.assert_called_once()
    # Assert that upsert was called via add_book
    assert mock_chroma_collection.upsert.call_count == 2
    mock_chroma_collection.upsert.assert_has_calls([
        call(ids=[book_id_1], documents=[f"Title 1. Abstract 1"], metadatas=[{'title': 'Title 1', 'description': 'Abstract 1'}]),
        call(ids=[book_id_2], documents=[f"Title 2"], metadatas=[{'title': 'Title 2', 'description': ''}]),
    ], any_order=True)
    mock_chroma_collection.delete.assert_not_called()

def test_sync_books_delete_removed_books_from_chroma(
    chroma_service_for_sync, mock_db_session, mock_book_service, mock_chroma_collection
):
    # Arrange
    book_id_1 = str(uuid.uuid4())
    book_id_2 = str(uuid.uuid4())
    mock_book_service.get_books.return_value = [] # Main DB is empty
    mock_chroma_collection.get.return_value = {"ids": [book_id_1, book_id_2]}

    # Act
    result = chroma_service_for_sync.sync_books()

    # Assert
    assert result == {"upserted": 0, "deleted": 2}
    mock_book_service.get_books.assert_called_once_with(limit=None)
    mock_chroma_collection.get.assert_called_once()
    mock_chroma_collection.upsert.assert_not_called()
    # Check that delete was called with the correct IDs (order doesn't matter)
    mock_chroma_collection.delete.assert_called_once()
    called_ids = mock_chroma_collection.delete.call_args.kwargs['ids']
    assert set(called_ids) == {book_id_1, book_id_2}

def test_sync_books_mix_of_adds_updates_deletions(
    chroma_service_for_sync, mock_db_session, mock_book_service, mock_chroma_collection
):
    # Arrange
    book_id_add = str(uuid.uuid4())
    book_id_update = str(uuid.uuid4())
    book_id_delete = str(uuid.uuid4())

    # DB state: one new book, one to be updated
    db_books = [
        create_mock_book(book_id_add, "New Book", "New Abstract"),
        create_mock_book(book_id_update, "Existing Book Updated", "Updated Abstract"),
    ]
    mock_book_service.get_books.return_value = db_books

    # ChromaDB state: one book to be updated, one to be deleted
    mock_chroma_collection.get.return_value = {"ids": [book_id_update, book_id_delete]}

    # Act
    result = chroma_service_for_sync.sync_books()

    # Assert
    assert result == {"upserted": 2, "deleted": 1}
    mock_book_service.get_books.assert_called_once_with(limit=None)
    mock_chroma_collection.get.assert_called_once()

    # Verify upserts for added and updated books
    assert mock_chroma_collection.upsert.call_count == 2
    mock_chroma_collection.upsert.assert_has_calls([
        call(ids=[book_id_add], documents=["New Book. New Abstract"], metadatas=[{"title": "New Book", "description": "New Abstract"}]),
        call(ids=[book_id_update], documents=["Existing Book Updated. Updated Abstract"], metadatas=[{"title": "Existing Book Updated", "description": "Updated Abstract"}]),
    ], any_order=True)

    # Verify deletion
    mock_chroma_collection.delete.assert_called_once_with(ids=[book_id_delete])

def test_sync_books_with_limit_parameter(
    chroma_service_for_sync, mock_db_session, mock_book_service, mock_chroma_collection
):
    # Arrange
    db_books = [create_mock_book(str(uuid.uuid4()), f"Book {i}", f"Abstract {i}") for i in range(3)]
    mock_book_service.get_books.return_value = db_books[:2] # BookService honors the limit
    mock_chroma_collection.get.return_value = {"ids": [str(b.book_id) for b in db_books]} # Chroma has all books

    # Act
    result = chroma_service_for_sync.sync_books(limit=2)

    # Assert
    assert result == {"upserted": 2, "deleted": 1} # Upserts 2, deletes the one not in the limited set
    mock_book_service.get_books.assert_called_once_with(limit=2)
    mock_chroma_collection.delete.assert_called_once_with(ids=[str(db_books[2].book_id)])

def test_sync_books_exception_handling(
    chroma_service_for_sync, mock_db_session, mock_book_service
):
    # Arrange
    mock_book_service.get_books.side_effect = Exception("Database error")

    # Act & Assert
    with pytest.raises(Exception, match="Database error"):
        chroma_service_for_sync.sync_books()

    # Verify session was closed even after exception
    mock_db_session.close.assert_called_once()

# --- Tests for /books/search/sync-from-db Endpoint ---

def test_sync_from_db_endpoint_success(client, mock_chroma_service_in_route):
    # Arrange
    mock_instance = mock_chroma_service_in_route.return_value
    mock_instance.sync_books.return_value = {"upserted": 10, "deleted": 5}
    mock_instance.llm_provider = "OPENAI"
    
    # Act
    response = client.post("/books/search/sync-from-db?limit=50")
    
    # Assert
    assert response.status_code == 200
    assert response.json() == {
        "message": "ChromaDB synchronization completed using OPENAI. Upserted: 10 books, Deleted: 5 books."
    }
    mock_instance.sync_books.assert_called_once_with(limit=50)

def test_sync_from_db_endpoint_exception(client, mock_chroma_service_in_route):
    # Arrange
    mock_instance = mock_chroma_service_in_route.return_value
    mock_instance.sync_books.side_effect = Exception("Sync failed")
    
    # Act
    response = client.post("/books/search/sync-from-db")
    
    # Assert
    assert response.status_code == 500
    assert "Failed to synchronize ChromaDB: Sync failed" in response.json()["detail"]

def test_sync_from_db_endpoint_unauthorized(client):
    # Arrange: Override get_current_user to raise an exception (simulating unauthorized)
    from fastapi import HTTPException
    async def mock_unauthorized():
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    app.dependency_overrides[get_current_user] = mock_unauthorized
    
    # Act
    response = client.post("/books/search/sync-from-db")
    
    # Assert
    assert response.status_code == 401
    assert response.json()["detail"] == "Not authenticated"
    
    # Clean up override for other tests
    del app.dependency_overrides[get_current_user]

# --- Tests for Initialization and Dependency Logic (get_chroma_service) ---

def test_get_chroma_service_fallback_to_ollama(client, mock_chroma_service_in_route):
    # Arrange: Mock env vars so OPENAI is default but key is missing
    with patch.dict(os.environ, {"LLM_PROVIDER": "OPENAI", "OPENAI_API_KEY": ""}):
        # Act
        client.post("/books/search/sync-from-db")
        
        # Assert: Check if ChromaService was instantiated with OLLAMA
        mock_chroma_service_in_route.assert_called()
        args, kwargs = mock_chroma_service_in_route.call_args
        assert kwargs['llm_provider_override'] == "OLLAMA"

def test_get_chroma_service_explicit_ollama(client, mock_chroma_service_in_route):
    # Arrange: Mock env vars for OLLAMA
    with patch.dict(os.environ, {"LLM_PROVIDER": "OLLAMA"}):
        # Act
        client.post("/books/search/sync-from-db")
        
        # Assert
        mock_chroma_service_in_route.assert_called()
        args, kwargs = mock_chroma_service_in_route.call_args
        assert kwargs['llm_provider_override'] == "OLLAMA"

def test_get_chroma_service_initialization_failure(client):
    # Arrange: Mock ChromaService to raise an exception during instantiation
    with patch('app.routes.chroma.ChromaService', side_effect=Exception("Init failed")):
        # Act
        response = client.post("/books/search/sync-from-db")
        
        # Assert
        assert response.status_code == 500
        assert "Failed to initialize ChromaDB service: Init failed" in response.json()["detail"]

# --- Tests for ChromaService Constructor Conflict Handling ---

def test_chroma_service_init_conflict_resets_collection(mock_chroma_client):
    # Arrange
    mock_collection = Mock()
    mock_chroma_client.get_or_create_collection.side_effect = ValueError("Embedding function conflict. persisted: OLLAMA, requested: OPENAI")
    mock_chroma_client.create_collection.return_value = mock_collection
    
    # Mock _initialize_llm_clients BUT ensure embedding_function exists
    def mock_init_clients(self):
        self.embedding_function = Mock()

    with patch.object(ChromaService, '_initialize_llm_clients', autospec=True, side_effect=mock_init_clients):
        with patch.dict(os.environ, {"LLM_PROVIDER": "OPENAI"}):
            # Act
            service = ChromaService()
            
            # Assert
            mock_chroma_client.delete_collection.assert_called_once_with(name="books")
            mock_chroma_client.create_collection.assert_called_once_with(
                name="books", embedding_function=service.embedding_function
            )
            assert service.collection == mock_collection

def test_chroma_service_init_other_value_error_re_raised(mock_chroma_client):
    # Arrange
    mock_chroma_client.get_or_create_collection.side_effect = ValueError("Some other error")
    
    def mock_init_clients(self):
        self.embedding_function = Mock()

    with patch.object(ChromaService, '_initialize_llm_clients', autospec=True, side_effect=mock_init_clients):
        # Act & Assert
        with pytest.raises(ValueError, match="Some other error"):
            ChromaService()
