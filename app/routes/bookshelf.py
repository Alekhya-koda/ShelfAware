# app/routes/bookshelf.py

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from typing import Optional, Literal
from pydantic import ValidationError

from app.dependencies.db import get_db
from app.dependencies.auth import get_current_db_user

from app.services.bookshelf_service import BookshelfService
from app.schemas.bookshelf import (
    BookshelfCreate,
    BookshelfRead,
    BookshelfStatusUpdate,
)

router = APIRouter()


def get_bookshelf_service(db: Session) -> BookshelfService:
    return BookshelfService(db)


def _extract_user_id(current_user) -> str:
    """
    get_current_user sometimes returns:
    - a dict (e.g. {"user_id": "...", "email": "..."}), OR
    - a User/Pydantic object with attribute .user_id

    This helper makes your routes robust to both.
    """
    if current_user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")

    # dict case
    if isinstance(current_user, dict):
        user_id = current_user.get("user_id") or current_user.get("sub") or current_user.get("id")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token payload (missing user_id)")
        return str(user_id)

    # object case
    user_id = getattr(current_user, "user_id", None) or getattr(current_user, "sub", None) or getattr(current_user, "id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid user object (missing user_id)")
    return str(user_id)


# 1) Add a book to shelf
@router.post("/", response_model=BookshelfRead, status_code=201)
def add_book(
    payload: BookshelfCreate,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_db_user),
):
    service = get_bookshelf_service(db)
    user_id = _extract_user_id(current_user)

    try:
        return service.add_to_shelf(user_id=user_id, book_id=payload.book_id)
    except ValueError as e:
        msg = str(e)
        if msg == "Book not found":
            raise HTTPException(status_code=404, detail=msg)
        if msg == "DUPLICATE":
            raise HTTPException(status_code=409, detail="Book already exists on shelf")
        raise HTTPException(status_code=400, detail=msg)


# 4) List "My shelves" with filters + sorting
@router.get("/", response_model=list[BookshelfRead])
def list_my_shelf(
    status: Optional[Literal["want_to_read", "currently_reading", "read"]] = Query(
        default=None, description="Filter by shelf status"
    ),
    sort: Literal["date_added", "updated_at", "date_finished"] = Query(
        default="updated_at", description="Sort field"
    ),
    order: Literal["asc", "desc"] = Query(default="desc", description="Sort order"),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_db_user),
):
    service = get_bookshelf_service(db)
    user_id = _extract_user_id(current_user)

    return service.list_shelf(user_id=user_id, status=status, sort=sort, order=order)


# 2) Remove a book from shelf
@router.delete("/{book_id}", status_code=204)
def remove_book(
    book_id: str,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_db_user),
):
    service = get_bookshelf_service(db)
    user_id = _extract_user_id(current_user)

    try:
        service.remove_from_shelf(user_id=user_id, book_id=book_id)
        return None
    except ValueError as e:
        msg = str(e)
        if msg == "NOT_FOUND":
            raise HTTPException(status_code=404, detail="Book not found on shelf")
        raise HTTPException(status_code=400, detail=msg)


# 3) Change shelf status + auto-fill started/finished dates
@router.patch("/{book_id}/status", response_model=BookshelfRead)
def update_status(
    book_id: str,
    payload: BookshelfStatusUpdate,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_db_user),
):
    service = get_bookshelf_service(db)
    user_id = _extract_user_id(current_user)

    try:
        return service.update_status(user_id=user_id, book_id=book_id, new_status=payload.shelf_status)
    except ValueError as e:
        msg = str(e)
        if msg == "NOT_FOUND":
            raise HTTPException(status_code=404, detail="Book not found on shelf")
        raise HTTPException(status_code=400, detail=msg)


# 5) Timeline
@router.get("/timeline", response_model=list[BookshelfRead])
def timeline(
    db: Session = Depends(get_db),
    current_user=Depends(get_current_db_user),
):
    service = get_bookshelf_service(db)
    user_id = _extract_user_id(current_user)
    return service.get_timeline(user_id=user_id)


# 6) Stats
@router.get("/stats")
def stats(
    db: Session = Depends(get_db),
    current_user=Depends(get_current_db_user),
):
    service = get_bookshelf_service(db)
    user_id = _extract_user_id(current_user)
    return service.get_stats(user_id=user_id)
