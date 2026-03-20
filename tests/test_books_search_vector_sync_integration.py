import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import os

from app.main import app
from app.services.chroma_service import ChromaService
from app.dependencies.auth import get_current_user
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.db.database import Base, get_db

# --- Fixtures for Route Testing ---

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

# --- Tests for /books/search/vector/sync Endpoint ---

def test_sync_from_db_endpoint_success(client, mock_chroma_service_in_route):
    # Arrange
    mock_instance = mock_chroma_service_in_route.return_value
    mock_instance.sync_books.return_value = {"upserted": 10, "deleted": 5}
    mock_instance.llm_provider = "OPENAI"
    
    # Act
    response = client.post("/books/search/vector/sync?limit=10")
    
    # Assert
    assert response.status_code == 200
    assert response.json() == {
        "message": "ChromaDB synchronization completed using OPENAI. Upserted: 10 books, Deleted: 5 books."
    }
    mock_instance.sync_books.assert_called_once_with(limit=10)

def test_sync_from_db_endpoint_exception(client, mock_chroma_service_in_route):
    # Arrange
    mock_instance = mock_chroma_service_in_route.return_value
    mock_instance.sync_books.side_effect = Exception("Sync failed")
    
    # Act
    response = client.post("/books/search/vector/sync")
    
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
    response = client.post("/books/search/vector/sync")
    
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
        client.post("/books/search/vector/sync")
        
        # Assert: Check if ChromaService was instantiated with OLLAMA
        mock_chroma_service_in_route.assert_called()
        args, kwargs = mock_chroma_service_in_route.call_args
        assert kwargs['llm_provider_override'] == "OLLAMA"

def test_get_chroma_service_explicit_ollama(client, mock_chroma_service_in_route):
    # Arrange: Mock env vars for OLLAMA
    with patch.dict(os.environ, {"LLM_PROVIDER": "OLLAMA"}):
        # Act
        client.post("/books/search/vector/sync")
        
        # Assert
        mock_chroma_service_in_route.assert_called()
        args, kwargs = mock_chroma_service_in_route.call_args
        assert kwargs['llm_provider_override'] == "OLLAMA"

def test_get_chroma_service_initialization_failure(client):
    # Arrange: Mock ChromaService to raise an exception during instantiation
    with patch('app.routes.chroma.ChromaService', side_effect=Exception("Init failed")):
        # Act
        response = client.post("/books/search/vector/sync")
        
        # Assert
        assert response.status_code == 500
        assert "Failed to initialize ChromaDB service: Init failed" in response.json()["detail"]

def test_sync_from_db_endpoint_query_param_override(client, mock_chroma_service_in_route):
    # Arrange
    mock_instance = mock_chroma_service_in_route.return_value
    mock_instance.sync_books.return_value = {"upserted": 0, "deleted": 0}
    mock_instance.llm_provider = "OLLAMA"
    
    # Act
    response = client.post("/books/search/vector/sync?llm_provider=OLLAMA")
    
    # Assert
    assert response.status_code == 200
    mock_chroma_service_in_route.assert_called_with(llm_provider_override="OLLAMA")
