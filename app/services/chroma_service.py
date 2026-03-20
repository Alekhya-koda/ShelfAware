import re
import chromadb
import logging
from chromadb.utils import embedding_functions
from typing import List, Optional, Literal # Added Literal
import os
from dotenv import load_dotenv

from sqlalchemy.orm import Session
from app.dependencies.db import get_db
from app.models.book import Book
from app.services.book_service import BookService

import openai
from ollama import Client as OllamaClient

load_dotenv()


class ChromaService:
    def __init__(self, llm_provider_override: Optional[Literal["OPENAI", "OLLAMA"]] = None): # Removed use_persisted_llm_provider parameter and is_retry
        # Initialize ChromaDB Persistent Client
        self.client = chromadb.PersistentClient(path="./chromadb")

        # Determine LLM provider from override, then environment variable, default to OPENAI
        self.llm_provider = (llm_provider_override or os.getenv("LLM_PROVIDER", "OPENAI")).upper()
        
        self._initialize_llm_clients() # Moved this call here
        
        try:
            # Initialize or create the "books" collection with the selected embedding function
            self.collection = self.client.get_or_create_collection(
                name="books",
                embedding_function=self.embedding_function
            )
        except ValueError as e:
            # Check if the error is specifically an embedding function conflict
            if "Embedding function conflict" in str(e):
                logging.warning(f"ChromaDB collection 'books' embedding function conflict detected: {e}")
                
                # Extract 'persisted' LLM provider from the error message
                match = re.search(r"persisted: (\w+)", str(e))
                persisted_llm_provider = match.group(1).upper() if match else "UNKNOWN"

                if self.llm_provider.upper() != persisted_llm_provider:
                    logging.warning(f"Requested LLM provider '{self.llm_provider}' is different from persisted provider '{persisted_llm_provider}'. Resetting ChromaDB collection to use '{self.llm_provider}'.")
                    # Delete the existing collection
                    self.client.delete_collection(name="books")
                    # Recreate the collection with the desired embedding function
                    self.collection = self.client.create_collection(
                        name="books",
                        embedding_function=self.embedding_function
                    )
                    logging.info(f"ChromaDB collection 'books' successfully reset and initialized with '{self.llm_provider}' embedding function.")
                else:
                    # If the requested provider is the same as persisted, but there's still a conflict (shouldn't happen)
                    # or if the conflict is some other ValueError that just happens to contain "Embedding function conflict"
                    # re-raise the original error.
                    logging.error(f"ChromaDB embedding function conflict with requested provider '{self.llm_provider}' which is same as persisted '{persisted_llm_provider}'. Re-raising original error.")
                    raise e
            else:
                # Re-raise if it's a different ValueError
                raise

    def _initialize_llm_clients(self): # New helper method
        """Helper method to initialize LLM embedding function and client based on self.llm_provider."""
        # Initialize embedding function and LLM client based on provider
        if self.llm_provider == "OPENAI":
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set for OpenAI provider.")
            self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                api_key=openai_api_key,
                model_name=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
            )
            self.llm_generator_client = openai.Client(api_key=openai_api_key)
            self.llm_model_for_generation = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
        elif self.llm_provider == "OLLAMA":
            ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
            self.embedding_function = embedding_functions.OllamaEmbeddingFunction(
                model_name=os.getenv("OLLAMA_EMBEDDING_MODEL", "embeddinggemma"),
                url=ollama_url
            )
            self.llm_generator_client = OllamaClient(host=ollama_url)
            self.llm_model_for_generation = os.getenv("OLLAMA_LLM_MODEL", "gemma3:1b")
        else:
            raise ValueError(f"Unsupported LLM_PROVIDER: {self.llm_provider}. Must be 'OLLAMA' or 'OPENAI'.")


    def add_book(self, book_id: str, title: str, abstract: Optional[str]): # Changed to abstract
        """
        Add a book's embedding to the collection.
        """
        # Adapted to handle Optional[str] for abstract
        document_content = f"{title}. {abstract}" if abstract else title
        self.collection.upsert(
            ids=[book_id],
            documents=[document_content],
            metadatas=[{"title": title, "description": abstract or ""}] # Use 'description' key for ChromaDB metadata
        )

    def search_books(self, query: str, n_results: int = 3, distance_threshold: float = 0.9) -> List[dict]:
        """
        Search for similar books based on a query with a similarity threshold.

        :param query: Query text for semantic search.
        :param n_results: Maximum number of results to retrieve.
        :param distance_threshold: Maximum similarity score to include a result.
        :return: List of metadata dictionaries for matching books.
        """
        results = self.collection.query(
            query_texts=[query], # Single query
            n_results=n_results
        )
        logging.info(f"Raw ChromaDB query results: {results}")

        # Flatten ids, distances and metadata (results are lists of lists)
        ids = results["ids"][0] if results["ids"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []
        logging.info(f"Extracted: {len(ids)} ids, {len(metadatas)} metadatas, {len(distances)} distances.")

        # Combine id, metadata and distances for filtering
        filtered_results = [
            {"id": book_id, **metadata, "distance": distance} # Add id and distance to each metadata entry
            for book_id, metadata, distance in zip(ids, metadatas, distances)
            if distance <= distance_threshold # Filter based on distance threshold
        ]
        logging.info(f"Filtered search results (distance <= {distance_threshold}): {filtered_results}")
        return filtered_results


    def generate_natural_language_response(self, query: str, search_results: List[dict]) -> str:
        """
        Generates a concise natural language summary of the search results using the configured LLM.
        """
        if not search_results:
            return f"No similar books found for the query: '{query}'."

        try:
            prompt_template = (
                    f'The user queried: "{query}". Below is a list of search results, where each item is a dictionary containing book information. '
                    f"Each dictionary has 'title' and 'description' keys. "
                    f"Your task is to summarize these {len(search_results)} books. "
                    f"For each book, identify its title and provide a brief, relevant summary of its description, highlighting aspects that directly relate to the user's query. "
                    f"Present the summary in a clear, easy-to-read natural language format, not as a list of dictionaries. The overall summary should be concise, ideally under 100 words. \n\n"
                    f"Search Results (Python list of dictionaries):\n{search_results}\n\n"
                    f"Please provide your concise summary now."
                )

            if self.llm_provider == "OPENAI":
                response = self.llm_generator_client.chat.completions.create(
                    model=self.llm_model_for_generation,
                    messages=[
                        {"role": "system", "content": "You are an expert librarian assistant. Your task is to provide concise and helpful summaries of book search results based on a user's query."},
                        {"role": "user", "content": prompt_template},
                    ],
                    max_tokens=200,
                    temperature=0.1
                )
                return response.choices[0].message.content
            elif self.llm_provider == "OLLAMA":
                response = self.llm_generator_client.chat(
                    model=self.llm_model_for_generation,
                    messages=[
                        {"role": "system", "content": "You are an expert librarian assistant. Your task is to provide concise and helpful summaries of book search results based on a user's query."},
                        {"role": "user", "content": prompt_template}
                    ],
                    options={
                        "temperature": 0.1,
                        "num_predict": 200,
                    }
                )
                return response['message']['content']
        except Exception as e:
            # Use specific error messages for clarity
            if self.llm_provider == "OPENAI":
                error_msg = f"Error generating summary with OpenAI: {str(e)}. Please ensure OPENAI_API_KEY is set and the model '{self.llm_model_for_generation}' is available."
            elif self.llm_provider == "OLLAMA":
                error_msg = f"Error generating summary with Ollama: {str(e)}. Please ensure Ollama is running and the model '{self.llm_model_for_generation}' is pulled."
            else:
                error_msg = f"Error generating summary with unsupported LLM_PROVIDER: {self.llm_provider} - {str(e)}."
            print(error_msg)
            return error_msg

    def delete_book(self, book_id: str):
        """
        Remove a book from the ChromaDB collection.
        """
        collection_name = self.collection.name
        logging.info(f"Attempting to delete book_id '{book_id}' from collection '{collection_name}'.")

        # Log state before deletion
        initial_items = self.collection.get()
        initial_ids = initial_items.get('ids', [])
        logging.info(f"Collection '{collection_name}' has {len(initial_ids)} items before deletion. IDs: {initial_ids}")

        self.collection.delete(ids=[book_id])

        # Log state after deletion
        final_items = self.collection.get()
        final_ids = final_items.get('ids', [])
        logging.info(f"Collection '{collection_name}' has {len(final_ids)} items after deletion. IDs: {final_ids}")

    def sync_books(self, limit: Optional[int] = None) -> dict:
        """
        Synchronizes books from the main database to ChromaDB.
        Handles additions, updates, and deletions to ensure ChromaDB
        reflects the current state of the main database.
        Returns a dictionary with counts of upserted and deleted books.
        
        :param limit: Optional. If provided, limits the number of books fetched from the main database.
                      Note: Without an explicit ORDER BY clause in the underlying query,
                      the selection of books when a limit is applied is not deterministic.
        """
        logging.info(f"Starting ChromaDB synchronization with limit: {limit if limit is not None else 'No limit'}...")
        db = next(get_db()) # Get a DB session
        book_service = BookService(db) # Instantiate BookService with the session

        upserted_count = 0
        deleted_count = 0

        try:
            # 1. Get all books from the main database
            logging.info("Fetching books from main database.")
            db_books = book_service.get_books(limit=limit) # Pass the limit to get_books
            db_book_ids = {str(book.book_id) for book in db_books}
            logging.info(f"Found {len(db_books)} books in the main database (limited by {limit if limit is not None else 'N/A'}).")

            # 2. Get all existing book IDs from ChromaDB
            logging.info("Fetching existing book IDs from ChromaDB.")
            chroma_collection_content = self.collection.get()
            chroma_book_ids = set(chroma_collection_content.get('ids', []))
            logging.info(f"Found {len(chroma_book_ids)} books in ChromaDB.")

            # 3. Add/Update books in ChromaDB (upsert)
            logging.info("Upserting books into ChromaDB...")
            for book in db_books:
                self.add_book(str(book.book_id), book.title, book.abstract)
                upserted_count += 1
            logging.info(f"Successfully upserted {upserted_count} books into ChromaDB.")

            # 4. Identify and delete books from ChromaDB that are no longer in the main DB
            books_to_delete_from_chroma = chroma_book_ids - db_book_ids
            if books_to_delete_from_chroma:
                logging.info(f"Deleting {len(books_to_delete_from_chroma)} books from ChromaDB no longer present in main database.")
                self.collection.delete(ids=list(books_to_delete_from_chroma))
                deleted_count = len(books_to_delete_from_chroma)
            else:
                logging.info("No books to delete from ChromaDB.")

            logging.info(f"ChromaDB synchronization completed. Upserted: {upserted_count}, Deleted: {deleted_count}.")
            return {"upserted": upserted_count, "deleted": deleted_count}

        except Exception as e:
            logging.error(f"Error during ChromaDB synchronization: {e}", exc_info=True)
            raise # Re-raise the exception after logging

        finally:
            db.close() # Always close the session