from pydantic import BaseModel, Field, ConfigDict    
from typing import Optional, Literal, Dict
from datetime import datetime, date


ShelfStatus = Literal["want_to_read", "currently_reading", "read"]


class BookshelfCreate(BaseModel):
    book_id: str


class BookshelfStatusUpdate(BaseModel):
    shelf_status: ShelfStatus


class BookshelfRead(BaseModel):
    user_id: str
    book_id: str
    shelf_status: ShelfStatus
    date_added: datetime
    date_started: Optional[datetime] = None
    date_finished: Optional[datetime] = None
    updated_at: datetime

    # Optional field if you later want to return saved synopsis from Bookshelf row
    synopsis: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class BookshelfTimelineItem(BaseModel):
    """
    Lightweight timeline view.
    You can expand this later to include book title/author by joining Book.
    """
    book_id: str
    shelf_status: ShelfStatus
    date_started: Optional[datetime] = None
    date_finished: Optional[datetime] = None
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class BookshelfStats(BaseModel):
    read_this_month: int
    read_this_year: int
    avg_days_to_finish: Optional[float] = None
    current_streak_days: int = 0
    best_streak_days: int = 0
