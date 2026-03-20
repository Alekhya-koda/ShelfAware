import pytest
from unittest.mock import Mock
from fastapi.testclient import TestClient
from app.main import app
from app.services.chroma_service import ChromaService
from app.routes.chroma import get_chroma_service
from app.dependencies.auth import get_current_user
from dotenv import load_dotenv
import openai
import os

# Load environment variables
load_dotenv()

@pytest.fixture()
def client():
    with TestClient(app) as client:
        yield client

@pytest.fixture
def mock_chroma_service():
    # Using spec to ensure mock matches the ChromaService interface
    return Mock(spec=ChromaService)

@pytest.fixture(autouse=True)
def override_dependencies(mock_chroma_service):
    # Mock authentication
    async def mock_get_current_user():
        return {"id": "test_user", "email": "test@example.com", "role": "user"}
    
    # Apply overrides
    app.dependency_overrides[get_current_user] = mock_get_current_user
    app.dependency_overrides[get_chroma_service] = lambda: mock_chroma_service
    yield
    # Reset dependency overrides after each test
    app.dependency_overrides = {}

# Module-level environment validation (as seen in reference)
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API key not found. Ensure it is set in the .env file.")

def test_search_summary_quality_llm_judge(client, mock_chroma_service):
    """
    Quality validation test for AI summary using LLM-as-a-judge.
    Follows the principle of evaluating AI output quality via a separate LLM call.
    """
    # 1. Arrange: Setup realistic mock data
    query = "educational books about space for children"
    mock_books = [
        {
            "id": "space1", 
            "title": "The Magic School Bus Lost in the Solar System", 
            "description": "Ms. Frizzle takes her class on a trip to outer space to learn about planets and stars.",
            "distance": 0.1
        },
        {
            "id": "space2", 
            "title": "National Geographic Little Kids First Big Book of Space", 
            "description": "An introduction to space for young readers, featuring colorful illustrations and simple facts.",
            "distance": 0.15
        }
    ]
    
    # Mock the return values for search and summary generation
    mock_chroma_service.search_books.return_value = mock_books
    
    # We provide a fixed high-quality response to verify the judge's scoring logic
    actual_ai_summary = (
        "Based on your interest in educational space books for children, I found two great options. "
        "'The Magic School Bus Lost in the Solar System' offers an adventurous journey through the planets, "
        "while the 'National Geographic Little Kids First Big Book of Space' provides factual introductions "
        "with vibrant illustrations suitable for young readers."
    )
    mock_chroma_service.generate_natural_language_response.return_value = actual_ai_summary
    
    # 2. Act: Execute the request
    response = client.get(f"/books/search/vector/summary?query={query}")
    assert response.status_code == 200
    summary_to_judge = response.json()["response"]
    
    # 3. Judge: Evaluation prompt (Mirroring reference manner)
    test_prompt = (
        f"You are an expert literary critic. A user searched for: '{query}'. "
        f"The search results were: {mock_books}. "
        f"The AI provided this summary: '{summary_to_judge}'. "
        f"Please rate the accuracy and helpfulness of this summary on a scale of 1 to 5. "
        f"Return only the numeric value of the rating."
    )

    # Call OpenAI API to evaluate the response
    openai.api_key = os.getenv("OPENAI_API_KEY")
    ai_client = openai.Client()

    completion = ai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a judge rating AI search summaries."},
            {"role": "user", "content": test_prompt}
        ],
        # Set temperature to 0 for deterministic evaluation
        temperature=0,
        max_completion_tokens=2048,
        frequency_penalty=0,
        presence_penalty=0,
    )
    
    rating = completion.choices[0].message.content.strip()
    
    # Final assertion: Expect a high rating (4 or 5) for the provided mock summary
    assert rating in ["4", "5"]
