import uuid, pydantic
from datetime import datetime, date
from sqlalchemy import Column, String, DateTime, Integer, Date
from sqlalchemy.orm import relationship
from app.db.database import Base

def new_uuid():
    return str(uuid.uuid4())

class Book(Base):
    __tablename__ = "book"

    book_id = Column(String, primary_key=True, default=new_uuid)
    title = Column(String, nullable=False)

    subtitle = Column(String, nullable=True)
    cover_image_url = Column(String, nullable=True)

    abstract = Column(String, nullable=True)
    CommunitySynopsis = Column(String, nullable=True)
    # JSON/stringified emotion profile created from reviews
    emotion_profile = Column(String, nullable=True)

    page_count = Column(Integer, nullable=True)
    published_date = Column(Date, nullable=True)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationship with bookshelves
    bookshelves = relationship("Bookshelf",  back_populates="book",  cascade="all, delete-orphan")

    # Relationship with reviews
    reviews = relationship("Review", back_populates="book", cascade="all, delete-orphan")

    #Relationship with genres
    genres = relationship("Genre", secondary="book_genre", back_populates="books", viewonly=True, lazy='selectin')

    #Relationship with book_genre
    book_genres = relationship("BookGenre", back_populates="book", cascade="all, delete-orphan")        
    
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional

class BookBase(BaseModel):
    title: str
    subtitle: Optional[str] = None
    cover_image_url: Optional[str] = None
    abstract: Optional[str] = None
    page_count: Optional[int] = None
    published_date: Optional[date] = None
    CommunitySynopsis: Optional[str] = None

class BookCreate(BookBase):
    pass

class BookUpdate(BaseModel):
    title: Optional[str] = None
    subtitle: Optional[str] = None
    cover_image_url: Optional[str] = None
    abstract: Optional[str] = None
    page_count: Optional[int] = None
    published_date: Optional[date] = None

class BookRead(BookBase):
    book_id: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)