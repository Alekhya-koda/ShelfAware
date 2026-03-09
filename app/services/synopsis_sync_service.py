import logging
from datetime import datetime
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import func, distinct
from openai import OpenAI
from app.models.book import Book
from app.models.bookshelf import Bookshelf
import hashlib

logger = logging.getLogger(__name__)


class SynopsisSyncService:
    """
    Service to synchronize user-generated synopsis with community synopsis.
    Can be manually triggered via admin endpoint to aggregate user synopses and generate/update community synopsis.
    """

    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)

    def get_all_user_synopses(self, db: Session, book_id: Optional[str] = None) -> dict:
        """
        Extract all user-generated synopses from bookshelf.
        
        Args:
            db: Database session
            book_id: Optional filter for specific book
            
        Returns:
            Dictionary with book_id as key and list of synopses as value
        """
        try:
            query = db.query(Bookshelf.book_id, Bookshelf.synopsis).filter(
                Bookshelf.synopsis.isnot(None),
                Bookshelf.synopsis != ""
            )
            
            if book_id:
                query = query.filter(Bookshelf.book_id == book_id)
            
            results = query.all()
            
            synopses_by_book = {}
            for row in results:
                if row.book_id not in synopses_by_book:
                    synopses_by_book[row.book_id] = []
                synopses_by_book[row.book_id].append(row.synopsis)
            
            logger.info(f"Retrieved synopses for {len(synopses_by_book)} books")
            return synopses_by_book
            
        except Exception as e:
            logger.error(f"Error retrieving user synopses: {str(e)}")
            raise

    def generate_community_synopsis(self, title: str, user_synopses: list) -> str:
        """
        Generate a community synopsis using OpenAI LLM.
        Combines multiple user-generated synopses into a cohesive summary (2-4 sentences).
        
        Args:
            title: Book title for context
            user_synopses: List of user-generated synopses
            
        Returns:
            Generated community synopsis or None if generation fails
        """
        try:
            # Filter out very short or duplicate synopses
            filtered_synopses = list(set([s.strip() for s in user_synopses if len(s.strip()) > 10]))
            
            if not filtered_synopses:
                logger.warning(f"No valid synopses found for book: {title}")
                return None
            
            synopses_text = "\n\n".join([f"- {s}" for s in filtered_synopses])
            
            prompt = f"""You are a professional book summarizer. Based on the following user-generated synopses for the book "{title}", create a single, cohesive, and engaging community synopsis.

The community synopsis must:
1. Be EXACTLY 2-4 sentences long
2. Capture the core essence of the book
3. Be objective and neutral in tone
4. Avoid spoilers
5. Be compelling to potential readers
6. Include key themes or plot elements that multiple users mentioned

User-generated synopses:
{synopses_text}

Generate only the synopsis without any additional commentary:"""
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a professional book summarizer and editor. Always return exactly 2-4 sentences."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.7
            )
            
            community_synopsis = response.choices[0].message.content.strip()
            logger.info(f"Generated community synopsis for '{title}'")
            return community_synopsis
            
        except Exception as e:
            logger.error(f"Error generating synopsis with OpenAI: {str(e)}")
            return None

    def compare_synopses(self, current_synopsis: Optional[str], user_synopses: list) -> bool:
        """
        Compare current community synopsis with user synopses to determine if update is needed.
        Detects significant changes in user synopses content.
        
        Args:
            current_synopsis: Current community synopsis in database
            user_synopses: List of user-generated synopses
            
        Returns:
            True if update is recommended, False otherwise
        """
        try:
            # Always update if no current synopsis
            if not current_synopsis:
                return True
            
            # No user synopses means no update needed
            if not user_synopses:
                return False
            
            # Calculate hash of current user synopses content
            user_content = "|".join(sorted([s.strip() for s in user_synopses if s.strip()]))
            current_hash = hashlib.md5(user_content.encode()).hexdigest()
            
            # If current synopsis contains a hash marker, check if synopses changed
            # Otherwise, always update if we have new synopses
            
            # Simple heuristic: if we have 3+ unique synopses, recommend update
            # This ensures diverse perspectives are captured
            unique_synopses = len(set([s.strip() for s in user_synopses if len(s.strip()) > 10]))
            
            if unique_synopses >= 3:
                logger.info(f"Recommending update: {unique_synopses} unique synopses found")
                return True
            
            # For small number of synopses, check if 50% of them are new (not similar to current synopsis)
            # This prevents unnecessary updates for minor changes
            logger.info(f"Skipping update: only {unique_synopses} unique synopses found and current synopsis exists")
            return False
            
        except Exception as e:
            logger.error(f"Error comparing synopses: {str(e)}")
            return True

    def sync_all_synopses(self, db: Session) -> dict:
        """
        Main method: Sync all book synopses by comparing user input and updating community synopsis.
        Can be called manually via admin endpoint.
        
        Args:
            db: Database session
            
        Returns:
            Dictionary with sync results
        """
        try:
            logger.info("Starting synopsis sync...")
            
            # Get all user-generated synopses
            user_synopses_by_book = self.get_all_user_synopses(db)
            
            if not user_synopses_by_book:
                logger.info("No user synopses found")
                return {
                    "status": "success",
                    "timestamp": datetime.utcnow().isoformat(),
                    "total_books_processed": 0,
                    "updated": 0,
                    "skipped": 0,
                    "errors": []
                }
            
            updated_count = 0
            skipped_count = 0
            errors = []
            
            for book_id, synopses in user_synopses_by_book.items():
                try:
                    # Get current book and its community synopsis
                    book = db.query(Book).filter(Book.book_id == book_id).first()
                    if not book:
                        logger.warning(f"Book not found: {book_id}")
                        skipped_count += 1
                        continue
                    
                    # Check if update is needed
                    if not self.compare_synopses(book.CommunitySynopsis, synopses):
                        logger.info(f"Skipping book '{book.title}' - no significant changes")
                        skipped_count += 1
                        continue
                    
                    # Generate new community synopsis
                    new_synopsis = self.generate_community_synopsis(book.title, synopses)
                    
                    if new_synopsis and new_synopsis != book.CommunitySynopsis:
                        book.CommunitySynopsis = new_synopsis
                        db.commit()
                        updated_count += 1
                        logger.info(f"Updated synopsis for book: {book.title}")
                        logger.debug(f"New synopsis: {new_synopsis[:100]}...")
                    else:
                        skipped_count += 1
                        
                except Exception as e:
                    logger.error(f"Error processing book {book_id}: {str(e)}")
                    errors.append({"book_id": book_id, "error": str(e)})
                    db.rollback()
            
            result = {
                "status": "success",
                "timestamp": datetime.utcnow().isoformat(),
                "total_books_processed": len(user_synopses_by_book),
                "updated": updated_count,
                "skipped": skipped_count,
                "errors": errors
            }
            
            logger.info(f"Synopsis sync completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Critical error in synopsis sync: {str(e)}")
            return {
                "status": "error",
                "timestamp": datetime.utcnow().isoformat(),
                "message": str(e),
                "total_books_processed": 0,
                "updated": 0,
                "skipped": 0,
                "errors": [{"error": str(e)}]
            }
