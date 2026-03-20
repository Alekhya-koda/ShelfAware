#Code 2
# app/schemas/book.py
from pydantic import BaseModel, ConfigDict, Field
from typing import Optional
from datetime import datetime, date

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
    #page_count: Optional[int] = None
    #newly added to replade statement above
    page_count: Optional[int] = Field(default=None, gt=0)
    published_date: Optional[date] = None
    CommunitySynopsis: Optional[str] = None

class BookRead(BookBase):
    book_id: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


#Code 1
'''
from pydantic import BaseModel, ConfigDict

class BookInfo(BaseModel):
    title: str
    author: str
    description: str | None = None

class BookResponse(BookInfo):
    model_config = ConfigDict(from_attributes=True)
    id: int
'''