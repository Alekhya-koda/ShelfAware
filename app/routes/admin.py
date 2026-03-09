import os
import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.services.cognito_service import RoleChecker, CognitoAdminRole
from app.services.synopsis_sync_service import SynopsisSyncService
from app.dependencies.db import get_db

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/users", dependencies=[Depends(RoleChecker("Admins"))])
def list_users():
    return {"message": "Admin access granted"}


@router.post("/sync-synopses")
async def sync_synopses_manual(db: Session = Depends(get_db)):
    """
    Manually trigger synopsis synchronization for all books.
    
    This endpoint:
    1. Retrieves all user-generated synopses from bookshelves
    2. Groups them by book_id
    3. For each book, determines if significant changes have occurred
    4. If changes are significant, generates a new community synopsis using OpenAI LLM
    5. Updates the CommunitySynopsis field in the books table
    
    Response:
    {
        "status": "success",
        "timestamp": "2026-03-09T...",
        "total_books_processed": 5,
        "updated": 3,
        "skipped": 2,
        "errors": []
    }
    """
    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.error("OPENAI_API_KEY environment variable not set")
            raise HTTPException(
                status_code=500,
                detail="OPENAI_API_KEY not configured. Cannot generate synopses."
            )
        
        logger.info("Manual synopsis sync initiated")
        
        # Initialize service and run sync
        service = SynopsisSyncService(openai_api_key=openai_api_key)
        result = service.sync_all_synopses(db)
        
        logger.info(f"Manual synopsis sync completed: {result}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during manual synopsis sync: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error during synopsis synchronization: {str(e)}"
        )