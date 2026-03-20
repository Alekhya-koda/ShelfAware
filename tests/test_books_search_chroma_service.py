import pytest
from unittest.mock import Mock, patch, call
import os
import uuid
from typing import Optional

from app.services.chroma_service import ChromaService
from app.models.book import Book
from app.services.book_service import BookService

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
def chroma_service_mocked(mock_chroma_client, mock_chroma_collection):
    # Bypass the original __init__ to avoid complex setup with LLM clients
    with patch.object(ChromaService, '__init__', lambda s, llm_provider_override=None: None):
        service = ChromaService()
        # Manually attach the mocks needed for the tests
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

# --- Tests for sync_books ---

def test_sync_books_no_books_in_db_or_chroma(
    chroma_service_mocked, mock_db_session, mock_book_service, mock_chroma_collection
):
    # Arrange
    mock_book_service.get_books.return_value = []
    mock_chroma_collection.get.return_value = {"ids": []}

    # Act
    result = chroma_service_mocked.sync_books()

    # Assert
    assert result == {"upserted": 0, "deleted": 0}
    mock_book_service.get_books.assert_called_once_with(limit=None)
    mock_chroma_collection.get.assert_called_once()
    mock_chroma_collection.upsert.assert_not_called()
    mock_chroma_collection.delete.assert_not_called()

def test_sync_books_add_new_books_to_chroma(
    chroma_service_mocked, mock_db_session, mock_book_service, mock_chroma_collection
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
    result = chroma_service_mocked.sync_books()

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
    chroma_service_mocked, mock_db_session, mock_book_service, mock_chroma_collection
):
    # Arrange
    book_id_1 = str(uuid.uuid4())
    book_id_2 = str(uuid.uuid4())
    mock_book_service.get_books.return_value = [] # Main DB is empty
    mock_chroma_collection.get.return_value = {"ids": [book_id_1, book_id_2]}

    # Act
    result = chroma_service_mocked.sync_books()

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
    chroma_service_mocked, mock_db_session, mock_book_service, mock_chroma_collection
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
    result = chroma_service_mocked.sync_books()

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
    chroma_service_mocked, mock_db_session, mock_book_service, mock_chroma_collection
):
    # Arrange
    db_books = [create_mock_book(str(uuid.uuid4()), f"Book {i}", f"Abstract {i}") for i in range(3)]
    mock_book_service.get_books.return_value = db_books[:2] # BookService honors the limit
    mock_chroma_collection.get.return_value = {"ids": [str(b.book_id) for b in db_books]} # Chroma has all books

    # Act
    result = chroma_service_mocked.sync_books(limit=2)

    # Assert
    assert result == {"upserted": 2, "deleted": 1} # Upserts 2, deletes the one not in the limited set
    mock_book_service.get_books.assert_called_once_with(limit=2)
    mock_chroma_collection.delete.assert_called_once_with(ids=[str(db_books[2].book_id)])

def test_sync_books_exception_handling(
    chroma_service_mocked, mock_db_session, mock_book_service
):
    # Arrange
    mock_book_service.get_books.side_effect = Exception("Database error")

    # Act & Assert
    with pytest.raises(Exception, match="Database error"):
        chroma_service_mocked.sync_books()

    # Verify session was closed even after exception
    mock_db_session.close.assert_called_once()

# --- Tests for search_books ---

def test_search_books_success(chroma_service_mocked):
    # Arrange
    mock_collection = chroma_service_mocked.collection
    mock_collection.query.return_value = {
        "ids": [["id1", "id2"]],
        "metadatas": [[{"title": "Book 1", "description": "Desc 1"}, {"title": "Book 2", "description": "Desc 2"}]],
        "distances": [[0.1, 0.2]]
    }
    
    # Act
    result = chroma_service_mocked.search_books("test query", n_results=2, distance_threshold=0.5)
    
    # Assert
    assert len(result) == 2
    assert result[0]["id"] == "id1"
    assert result[0]["title"] == "Book 1"
    assert result[0]["distance"] == 0.1
    mock_collection.query.assert_called_once_with(query_texts=["test query"], n_results=2)

def test_search_books_no_results(chroma_service_mocked):
    # Arrange
    mock_collection = chroma_service_mocked.collection
    mock_collection.query.return_value = {"ids": [], "metadatas": [], "distances": []}
    
    # Act
    result = chroma_service_mocked.search_books("test query")
    
    # Assert
    assert result == []

def test_search_books_filter_by_distance(chroma_service_mocked):
    # Arrange
    mock_collection = chroma_service_mocked.collection
    mock_collection.query.return_value = {
        "ids": [["id1", "id2", "id3"]],
        "metadatas": [
            [
                {"title": "Book 1", "description": "Desc 1"},
                {"title": "Book 2", "description": "Desc 2"},
                {"title": "Book 3", "description": "Desc 3"}
            ]
        ],
        "distances": [[0.3, 0.8, 0.1]]  # 0.8 > 0.5 threshold
    }
    
    # Act
    result = chroma_service_mocked.search_books("test query", distance_threshold=0.5)
    
    # Assert
    assert len(result) == 2  # id1 and id3
    assert result[0]["id"] == "id1"
    assert result[1]["id"] == "id3"

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

# --- Additional tests for Initialization Edge Cases and Provider Logic ---

def test_chroma_service_init_conflict_same_provider_re_raises(mock_chroma_client):
    # Arrange: Error says persisted is OPENAI, and we requested OPENAI
    mock_chroma_client.get_or_create_collection.side_effect = ValueError(
        "Embedding function conflict. persisted: OPENAI, requested: OPENAI"
    )
    
    with patch.object(ChromaService, '_initialize_llm_clients', lambda s: setattr(s, 'embedding_function', Mock())):
        with patch.dict(os.environ, {"LLM_PROVIDER": "OPENAI"}):
            # Act & Assert
            with pytest.raises(ValueError, match="persisted: OPENAI"):
                ChromaService()

def test_initialize_llm_clients_openai_no_key():
    # Arrange
    with patch.dict(os.environ, {"LLM_PROVIDER": "OPENAI", "OPENAI_API_KEY": ""}, clear=True):
        # We use __new__ to avoid calling __init__ which triggers the complex setup
        service = ChromaService.__new__(ChromaService)
        service.llm_provider = "OPENAI"
        
        # Act & Assert
        with pytest.raises(ValueError, match="OPENAI_API_KEY environment variable not set"):
            service._initialize_llm_clients()

def test_initialize_llm_clients_openai_config():
    # Arrange
    with patch.dict(os.environ, {
        "LLM_PROVIDER": "OPENAI",
        "OPENAI_API_KEY": "test-key",
        "OPENAI_EMBEDDING_MODEL": "test-emb",
        "OPENAI_LLM_MODEL": "test-llm"
    }):
        service = ChromaService.__new__(ChromaService)
        service.llm_provider = "OPENAI"
        
        with patch('app.services.chroma_service.embedding_functions.OpenAIEmbeddingFunction') as MockEmb, \
             patch('app.services.chroma_service.openai.Client') as MockClient:
            
            # Act
            service._initialize_llm_clients()
            
            # Assert
            MockEmb.assert_called_once_with(api_key="test-key", model_name="test-emb")
            MockClient.assert_called_once_with(api_key="test-key")
            assert service.llm_model_for_generation == "test-llm"

def test_initialize_llm_clients_ollama_config():
    # Arrange
    with patch.dict(os.environ, {
        "LLM_PROVIDER": "OLLAMA",
        "OLLAMA_URL": "http://test-ollama:11434",
        "OLLAMA_EMBEDDING_MODEL": "test-emb",
        "OLLAMA_LLM_MODEL": "test-llm"
    }):
        service = ChromaService.__new__(ChromaService)
        service.llm_provider = "OLLAMA"
        
        with patch('app.services.chroma_service.embedding_functions.OllamaEmbeddingFunction') as MockEmb, \
             patch('app.services.chroma_service.OllamaClient') as MockClient:
            
            # Act
            service._initialize_llm_clients()
            
            # Assert
            MockEmb.assert_called_once_with(model_name="test-emb", url="http://test-ollama:11434")
            MockClient.assert_called_once_with(host="http://test-ollama:11434")
            assert service.llm_model_for_generation == "test-llm"

def test_initialize_llm_clients_unsupported_provider():
    # Arrange
    service = ChromaService.__new__(ChromaService)
    service.llm_provider = "ANTHROPIC"
    
    # Act & Assert
    with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER: ANTHROPIC"):
        service._initialize_llm_clients()

# --- Additional tests for Sync Logic Branch Coverage ---

def test_sync_books_no_deletions_needed(
    chroma_service_mocked, mock_db_session, mock_book_service, mock_chroma_collection
):
    # Arrange: DB and Chroma have the exact same book
    book_id = str(uuid.uuid4())
    db_books = [create_mock_book(book_id, "Title", "Abstract")]
    mock_book_service.get_books.return_value = db_books
    mock_chroma_collection.get.return_value = {"ids": [book_id]}

    # Act
    result = chroma_service_mocked.sync_books()

    # Assert
    assert result == {"upserted": 1, "deleted": 0}
    mock_chroma_collection.delete.assert_not_called()

def test_sync_books_successful_session_closure(
    chroma_service_mocked, mock_db_session, mock_book_service, mock_chroma_collection
):
    # Arrange
    mock_book_service.get_books.return_value = []
    mock_chroma_collection.get.return_value = {"ids": []}

    # Act
    chroma_service_mocked.sync_books()

    # Assert
    mock_db_session.close.assert_called_once()

# --- Tests for add_book and delete_book ---

def test_add_book_success(chroma_service_mocked, mock_chroma_collection):
    # Act
    chroma_service_mocked.add_book("id1", "Title", "Abstract")
    
    # Assert
    mock_chroma_collection.upsert.assert_called_once_with(
        ids=["id1"],
        documents=["Title. Abstract"],
        metadatas=[{"title": "Title", "description": "Abstract"}]
    )

def test_add_book_no_abstract(chroma_service_mocked, mock_chroma_collection):
    # Act
    chroma_service_mocked.add_book("id1", "Title", None)
    
    # Assert
    mock_chroma_collection.upsert.assert_called_once_with(
        ids=["id1"],
        documents=["Title"],
        metadatas=[{"title": "Title", "description": ""}]
    )

def test_delete_book_success(chroma_service_mocked, mock_chroma_collection):
    # Arrange
    mock_chroma_collection.name = "books"
    mock_chroma_collection.get.side_effect = [{"ids": ["id1"]}, {"ids": []}]
    
    # Act
    chroma_service_mocked.delete_book("id1")
    
    # Assert
    mock_chroma_collection.delete.assert_called_once_with(ids=["id1"])
    assert mock_chroma_collection.get.call_count == 2

# --- Tests for search_books Edge Cases ---

def test_search_books_missing_metadata_keys(chroma_service_mocked):
    # Arrange
    mock_collection = chroma_service_mocked.collection
    mock_collection.query.return_value = {
        "ids": [["id1"]],
        "metadatas": [[{}]], # Empty metadata
        "distances": [[0.1]]
    }
    
    # Act
    result = chroma_service_mocked.search_books("test query")
    
    # Assert
    # The current code uses **metadata, which won't include title/description if missing
    assert result[0]["id"] == "id1"
    assert "title" not in result[0]
    assert "description" not in result[0]

# --- Tests for generate_natural_language_response ---

def test_generate_natural_language_response_no_results(chroma_service_mocked):
    # Act
    result = chroma_service_mocked.generate_natural_language_response("query", [])
    
    # Assert
    assert result == "No similar books found for the query: 'query'."

def test_generate_natural_language_response_openai_success(chroma_service_mocked):
    # Arrange
    chroma_service_mocked.llm_provider = "OPENAI"
    chroma_service_mocked.llm_model_for_generation = "gpt-4"
    mock_client = Mock()
    chroma_service_mocked.llm_generator_client = mock_client
    
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Mocked summary"))]
    mock_client.chat.completions.create.return_value = mock_response
    
    search_results = [{"title": "Book 1", "description": "Desc 1"}]
    
    # Act
    result = chroma_service_mocked.generate_natural_language_response("test query", search_results)
    
    # Assert
    assert result == "Mocked summary"
    mock_client.chat.completions.create.assert_called_once()

def test_generate_natural_language_response_ollama_success(chroma_service_mocked):
    # Arrange
    chroma_service_mocked.llm_provider = "OLLAMA"
    chroma_service_mocked.llm_model_for_generation = "gemma"
    mock_client = Mock()
    chroma_service_mocked.llm_generator_client = mock_client
    
    mock_client.chat.return_value = {"message": {"content": "Mocked Ollama summary"}}
    
    search_results = [{"title": "Book 1", "description": "Desc 1"}]
    
    # Act
    result = chroma_service_mocked.generate_natural_language_response("test query", search_results)
    
    # Assert
    assert result == "Mocked Ollama summary"
    mock_client.chat.assert_called_once()

def test_generate_natural_language_response_openai_error(chroma_service_mocked):
    # Arrange
    chroma_service_mocked.llm_provider = "OPENAI"
    chroma_service_mocked.llm_model_for_generation = "gpt-4"
    mock_client = Mock()
    chroma_service_mocked.llm_generator_client = mock_client
    mock_client.chat.completions.create.side_effect = Exception("OpenAI API error")
    
    search_results = [{"title": "Book 1", "description": "Desc 1"}]
    
    # Act
    result = chroma_service_mocked.generate_natural_language_response("test query", search_results)
    
    # Assert
    assert "Error generating summary with OpenAI: OpenAI API error" in result

def test_generate_natural_language_response_ollama_error(chroma_service_mocked):
    # Arrange
    chroma_service_mocked.llm_provider = "OLLAMA"
    chroma_service_mocked.llm_model_for_generation = "gemma"
    mock_client = Mock()
    chroma_service_mocked.llm_generator_client = mock_client
    mock_client.chat.side_effect = Exception("Ollama error")
    
    search_results = [{"title": "Book 1", "description": "Desc 1"}]
    
    # Act
    result = chroma_service_mocked.generate_natural_language_response("test query", search_results)
    
    # Assert
    assert "Error generating summary with Ollama: Ollama error" in result

def test_generate_natural_language_response_unsupported_provider_case(chroma_service_mocked):
    # Arrange
    chroma_service_mocked.llm_provider = "UNKNOWN"
    search_results = [{"title": "Book 1", "description": "Desc 1"}]
    
    # Act
    result = chroma_service_mocked.generate_natural_language_response("query", search_results)
    
    # Assert
    assert result is None

def test_generate_natural_language_response_unsupported_provider_exception(chroma_service_mocked):
    # Arrange
    chroma_service_mocked.llm_provider = "UNKNOWN"
    class ErrorRaiser:
        def __len__(self): raise RuntimeError("Custom error")
        def __bool__(self): return True
    
    # Act
    result = chroma_service_mocked.generate_natural_language_response("query", ErrorRaiser())
    
    # Assert
    assert "Error generating summary with unsupported LLM_PROVIDER: UNKNOWN - Custom error" in result
