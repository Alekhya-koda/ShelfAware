import os
import logging
from fastapi import APIRouter, HTTPException, status, Depends
from sqlalchemy.orm import Session
from app.dependencies.db import get_db
from app.dependencies.auth import get_current_user
from app.services.chroma_service import ChromaService
from app.schemas.chroma_book import ChromaBookInfo
from typing import Optional, Literal


router = APIRouter()

# Determine default LLM provider based on environment variables
# If LLM_PROVIDER is not set, it defaults to OPENAI.
# If OPENAI is the default and OPENAI_API_KEY is not set, switch to OLLAMA.

def get_chroma_service(
    llm_provider: Optional[Literal["OPENAI", "OLLAMA"]] = None
) -> ChromaService:
    # Determine default LLM provider based on environment variables
    # If LLM_PROVIDER is not set, it defaults to OPENAI.
    # If OPENAI is the default and OPENAI_API_KEY is not set, switch to OLLAMA.
    default_llm_provider_env = os.getenv("LLM_PROVIDER", "OPENAI").upper()
    llm_provider_for_instance = None
    if default_llm_provider_env == "OPENAI" and not os.getenv("OPENAI_API_KEY"):
        llm_provider_for_instance = "OLLAMA"
    elif default_llm_provider_env == "OLLAMA":
        llm_provider_for_instance = "OLLAMA"
    
    # If a specific llm_provider is provided to the dependency, use it.
    # Otherwise, fall back to the instance-level determination.
    provider_to_use = llm_provider or llm_provider_for_instance

    try:
        # Attempt to get ChromaService. Conflicts are handled internally now.
        return ChromaService(llm_provider_override=provider_to_use)
    except Exception as e:
        logging.error(f"Failed to initialize ChromaService: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize ChromaDB service: {str(e)}"
        )


@router.post("/vector/sync", status_code=status.HTTP_200_OK)
def sync_chromadb_from_db(
    limit: int = 30,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
    chroma_service: ChromaService = Depends(get_chroma_service),
    llm_provider: Optional[Literal["OPENAI", "OLLAMA"]] = None, # Now it's just a query param
):
    """
    Manually trigger synchronization of all books from the main database to ChromaDB.
    This ensures that the ChromaDB search index is up-to-date with the latest book records,
    handling additions, updates, and deletions.
    An optional `llm_provider` can be specified to override the default for this sync operation.
    """
    logging.info(f"Initiating ChromaDB synchronization, triggered by admin. LLM Provider: {llm_provider}")
    try:
        sync_results = chroma_service.sync_books(limit=limit) # Capture the results
        upserted = sync_results.get("upserted", 0)
        deleted = sync_results.get("deleted", 0)
        logging.info(f"ChromaDB synchronization completed. Upserted: {upserted} books, Deleted: {deleted} books.")
        return {"message": f"ChromaDB synchronization completed using {chroma_service.llm_provider}. Upserted: {upserted} books, Deleted: {deleted} books."}
    except Exception as e:
        logging.error(f"Failed to synchronize ChromaDB: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to synchronize ChromaDB: {str(e)}")


@router.get("/vector/search")
def search_books_in_chromadb(
    query: str,
    distance_threshold: float = 0.9,
    llm_provider: Optional[Literal["OPENAI", "OLLAMA"]] = None, # Keep as query param
    current_user: dict = Depends(get_current_user), # Added authentication
    chroma_service: ChromaService = Depends(get_chroma_service),
):
    """
    Search for similar books in ChromaDB based on a query and a distance_threshold.
    An optional `llm_provider` can be specified to override the default for this search operation.
    """
    results = chroma_service.search_books(query, distance_threshold=distance_threshold)
    if not results:
        raise HTTPException(status_code=404, detail=f"No similar books found for the query: '{query}'.")

    return {"query": query, "response": results}


@router.get("/vector/summary")
def ai_search_books_in_chromadb(
    query: str,
    distance_threshold: float = 0.9,
    llm_provider: Optional[Literal["OPENAI", "OLLAMA"]] = None, # Keep as query param
    current_user: dict = Depends(get_current_user), # Added authentication
    chroma_service: ChromaService = Depends(get_chroma_service),
):
    """
    Search for similar books in ChromaDB based on a query and return a natural language summary.
    An optional `llm_provider` can be specified to override the default for this summary generation.
    """
    results = chroma_service.search_books(query, distance_threshold=distance_threshold)
    if not results:
        raise HTTPException(status_code=404, detail=f"No similar books found for the query: '{query}'.")

    # Generate a natural language response using the selected LLM provider
    response = chroma_service.generate_natural_language_response(query, results)
    return {"query": query, "response": response}

@router.delete("/vector/{book_id}")
def delete_book(
    book_id: str,
    current_user: dict = Depends(get_current_user), # Added authentication
    chroma_service: ChromaService = Depends(get_chroma_service),
):
    """
    Delete a book's vector and metadata from ChromaDB by its ID.
    
    This endpoint also highlights a task from Activity 3: Test and improve the implementation of the delete_book method.
    """
    try:
        chroma_service.delete_book(book_id)
        return {"message": f"Book with ID '{book_id}' deleted from ChromaDB successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete book from ChromaDB: {str(e)}")
