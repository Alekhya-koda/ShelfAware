import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import os

from app.main import app
from app.services.chroma_service import ChromaService
from app.dependencies.auth import get_current_user

# Mock environment variables
@pytest.fixture(autouse=True)
def mock_env_vars():
    with patch.dict(os.environ, {
        "OPENAI_API_KEY": "test_openai_key",
        "OPENAI_EMBEDDING_MODEL": "test_embedding_model",
        "OPENAI_LLM_MODEL": "test_llm_model",
        "LLM_PROVIDER": "OPENAI"
    }):
        yield

# Mock ChromaService for route testing
@pytest.fixture
def mock_chroma_service():
    with patch('app.routes.chroma.ChromaService') as MockChromaService:
        mock_instance = Mock(spec=ChromaService)
        mock_instance.llm_provider = "OPENAI"
        MockChromaService.return_value = mock_instance
        yield MockChromaService

# Test client with mocked auth
@pytest.fixture
def client():
    async def mock_get_current_user():
        return {"id": "test_user", "email": "test@example.com", "role": "user"}
    
    app.dependency_overrides[get_current_user] = mock_get_current_user
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()

# --- Integration Tests for /books/search/vector/summary Endpoint ---

def test_search_summary_endpoint_success(client, mock_chroma_service):
    # Arrange
    mock_instance = mock_chroma_service.return_value
    mock_instance.search_books.return_value = [
        {"id": "1", "title": "Test Book", "description": "Test Desc", "distance": 0.1}
    ]
    mock_instance.generate_natural_language_response.return_value = "This is an AI generated summary of the search results."
    
    # Act
    response = client.get("/books/search/vector/summary?query=test+book")
    
    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "test book"
    assert "summary" in data["response"].lower() or "ai" in data["response"].lower()
    assert data["response"] == "This is an AI generated summary of the search results."
    mock_instance.generate_natural_language_response.assert_called_once()

def test_search_summary_endpoint_no_results(client, mock_chroma_service):
    # Arrange
    mock_instance = mock_chroma_service.return_value
    mock_instance.search_books.return_value = []
    
    # Act
    response = client.get("/books/search/vector/summary?query=nonexistent")
    
    # Assert
    assert response.status_code == 404
    assert "No similar books found" in response.json()["detail"]

def test_search_summary_endpoint_with_parameters(client, mock_chroma_service):
    # Arrange
    mock_instance = mock_chroma_service.return_value
    mock_instance.search_books.return_value = [
        {"id": "1", "title": "Book", "description": "Desc", "distance": 0.2}
    ]
    mock_instance.generate_natural_language_response.return_value = "Summary"
    
    # Act
    response = client.get("/books/search/vector/summary?query=test&distance_threshold=0.8&llm_provider=OLLAMA")
    
    # Assert
    assert response.status_code == 200
    mock_instance.search_books.assert_called_once_with("test", distance_threshold=0.8)

def test_search_summary_endpoint_unauthorized(client):
    # Arrange: Override to simulate unauthorized
    from fastapi import HTTPException
    async def mock_unauthorized():
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    app.dependency_overrides[get_current_user] = mock_unauthorized
    
    # Act
    response = client.get("/books/search/vector/summary?query=test")
    
    # Assert
    assert response.status_code == 401
    app.dependency_overrides.clear()
