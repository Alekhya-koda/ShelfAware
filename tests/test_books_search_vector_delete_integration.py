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

# --- Integration Tests for DELETE /books/search/vector/{book_id} Endpoint ---

def test_delete_book_endpoint_success(client, mock_chroma_service):
    # Arrange
    mock_instance = mock_chroma_service.return_value
    mock_instance.delete_book.return_value = None
    
    # Act
    response = client.delete("/books/search/vector/test_id")
    
    # Assert
    assert response.status_code == 200
    assert response.json() == {"message": "Book with ID 'test_id' deleted from ChromaDB successfully."}
    mock_instance.delete_book.assert_called_once_with("test_id")

def test_delete_book_endpoint_failure(client, mock_chroma_service):
    # Arrange
    mock_instance = mock_chroma_service.return_value
    mock_instance.delete_book.side_effect = Exception("Delete failed")
    
    # Act
    response = client.delete("/books/search/vector/error_id")
    
    # Assert
    assert response.status_code == 500
    assert "Failed to delete book from ChromaDB: Delete failed" in response.json()["detail"]

def test_delete_book_endpoint_unauthorized(client):
    # Arrange: Override to simulate unauthorized
    from fastapi import HTTPException
    async def mock_unauthorized():
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    app.dependency_overrides[get_current_user] = mock_unauthorized
    
    # Act
    response = client.delete("/books/search/vector/test_id")
    
    # Assert
    assert response.status_code == 401
    app.dependency_overrides.clear()
