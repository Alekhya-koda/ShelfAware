from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.orm import Session
from app.schemas.book import BookCreate, BookUpdate, BookRead
from app.services.book_service import BookService
from app.dependencies.services import get_book_service
from app.dependencies.roles import required_admin_role
from app.dependencies.db import get_db
from app.models.genre import Genre

router = APIRouter()

@router.get("/", response_model=list[BookRead])
def get_books(service: BookService = Depends(get_book_service)):
    return service.get_books()


@router.get("/genres", response_model=list[str])
def get_genres(db: Session = Depends(get_db)):
    rows = db.query(Genre.name).order_by(Genre.name.asc()).all()
    return [name for (name,) in rows]

@router.get("/{book_id}", response_model=BookRead)
def get_book(book_id: str, service: BookService = Depends(get_book_service)):
    book = service.get_book(book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    return book

# ✅ Admin only
@router.post(
    "/",
    response_model=BookRead,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(required_admin_role)],  # 👈 admin guard
)
def add_book(book: BookCreate, service: BookService = Depends(get_book_service)):
    return service.add_book(book)

# ✅ Admin only
@router.put(
    "/{book_id}",
    response_model=BookRead,
    dependencies=[Depends(required_admin_role)],  # 👈 admin guard
)
def update_book(book_id: str, updated_book: BookUpdate, service: BookService = Depends(get_book_service)):
    book = service.update_book(book_id, updated_book)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    return book

# ✅ Admin only
@router.delete(
    "/{book_id}",
    dependencies=[Depends(required_admin_role)],  # 👈 admin guard
)
def delete_book(book_id: str, service: BookService = Depends(get_book_service)):
    success = service.delete_book(book_id)
    if not success:
        raise HTTPException(status_code=404, detail="Book not found")
    return {"message": "Book deleted successfully"}