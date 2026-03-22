# app/services/bookshelf_service.py

from datetime import datetime
from typing import Optional, Dict, Any, List, Literal
import json

from sqlalchemy.orm import Session
from sqlalchemy import select, desc, asc, or_

from app.models.bookshelf import Bookshelf
from app.models.book import Book  # keep for existence check


STATUS_ORDER = {
    "want_to_read": 0,
    "currently_reading": 1,
    "read": 2,
}

SORT_MAP = {
    "date_added": Bookshelf.date_added,
    "updated_at": Bookshelf.updated_at,
    "date_finished": Bookshelf.date_finished,
}


def _now() -> datetime:
    # naive UTC for consistency
    return datetime.utcnow()


def _validate_transition(old: str, new: str) -> None:
    if new not in STATUS_ORDER:
        raise ValueError("Invalid shelf_status")
    if STATUS_ORDER[new] < STATUS_ORDER.get(old, 0):
        raise ValueError("Cannot move status backwards")


class BookshelfService:
    """
    Service layer for Bookshelf operations.

    Router expects:
      service = BookshelfService(db)
      service.add_to_shelf(...)
      service.list_shelf(...)
      service.remove_from_shelf(...)
      service.update_status(...)
      service.get_timeline(...)
      service.get_stats(...)
    """

    def __init__(self, db: Session):
        self.db = db

    def add_to_shelf(self, *, user_id: str, book_id: str) -> Bookshelf:
        # Optional: verify book exists (recommended)
        exists = self.db.execute(
            select(Book.book_id).where(Book.book_id == book_id)
        ).scalar_one_or_none()
        if not exists:
            raise ValueError("Book not found")

        existing = self.db.execute(
            select(Bookshelf).where(
                Bookshelf.user_id == user_id,
                Bookshelf.book_id == book_id,
            )
        ).scalar_one_or_none()

        if existing:
            # route can map this to 409 Conflict
            raise ValueError("DUPLICATE")

        now = _now()
        item = Bookshelf(
            user_id=user_id,
            book_id=book_id,
            shelf_status="want_to_read",
            date_added=now,
            updated_at=now,
        )
        self.db.add(item)
        self.db.commit()
        self.db.refresh(item)
        return item

    def remove_from_shelf(self, *, user_id: str, book_id: str) -> None:
        item = self.db.execute(
            select(Bookshelf).where(
                Bookshelf.user_id == user_id,
                Bookshelf.book_id == book_id,
            )
        ).scalar_one_or_none()

        if not item:
            raise ValueError("NOT_FOUND")

        self.db.delete(item)
        self.db.commit()

    def update_status(self, *, user_id: str, book_id: str, new_status: str) -> Bookshelf:
        item = self.db.execute(
            select(Bookshelf).where(
                Bookshelf.user_id == user_id,
                Bookshelf.book_id == book_id,
            )
        ).scalar_one_or_none()

        if not item:
            raise ValueError("NOT_FOUND")

        old_status = item.shelf_status
        _validate_transition(old_status, new_status)

        now = _now()
        item.shelf_status = new_status
        item.updated_at = now

        if new_status == "currently_reading":
            if item.date_started is None:
                item.date_started = now

        if new_status == "read":
            if item.date_started is None:
                item.date_started = now
            if item.date_finished is None:
                item.date_finished = now
            if item.date_finished < item.date_started:
                raise ValueError("Invalid dates: finished before started")

        self.db.commit()
        self.db.refresh(item)
        return item

    def update_progress(
        self,
        *,
        user_id: str,
        book_id: str,
        progress_percent: int,
        mood: Optional[str] = None,
        moods: Optional[List[str]] = None,
    ) -> Bookshelf:
        item = self.db.execute(
            select(Bookshelf).where(
                Bookshelf.user_id == user_id,
                Bookshelf.book_id == book_id,
            )
        ).scalar_one_or_none()

        if not item:
            raise ValueError("NOT_FOUND")

        if item.shelf_status not in ("currently_reading", "read"):
            raise ValueError("Progress can only be updated for currently reading or read books")

        now = _now()
        payload: Dict[str, Any] = {}
        if item.synopsis:
            try:
                payload = json.loads(item.synopsis)
                if not isinstance(payload, dict):
                    payload = {}
            except Exception:
                payload = {}

        normalized_moods: List[str] = []
        if moods:
            seen = set()
            for raw in moods:
                val = (raw or "").strip()
                if val and val.lower() not in seen:
                    seen.add(val.lower())
                    normalized_moods.append(val)
        elif mood:
            split_vals = [part.strip() for part in mood.split(",")]
            normalized_moods = [m for m in split_vals if m]

        payload["progress_percent"] = int(progress_percent)
        payload["moods"] = normalized_moods
        payload["mood"] = ", ".join(normalized_moods) if normalized_moods else None
        payload["last_check_in_at"] = now.isoformat()

        item.synopsis = json.dumps(payload)
        item.updated_at = now

        self.db.commit()
        self.db.refresh(item)
        return item

    def list_shelf(
        self,
        *,
        user_id: str,
        status: Optional[str] = None,
        sort: str = "updated_at",
        order: str = "desc",
    ) -> List[Bookshelf]:
        q = select(Bookshelf).where(Bookshelf.user_id == user_id)

        if status:
            q = q.where(Bookshelf.shelf_status == status)

        sort_col = SORT_MAP.get(sort, Bookshelf.updated_at)
        q = q.order_by(desc(sort_col) if order == "desc" else asc(sort_col))

        return self.db.execute(q).scalars().all()

    def get_timeline(self, *, user_id: str) -> List[Bookshelf]:
        q = (
            select(Bookshelf)
            .where(Bookshelf.user_id == user_id)
            .where(or_(Bookshelf.date_started.isnot(None), Bookshelf.date_finished.isnot(None)))
            .order_by(desc(Bookshelf.date_finished), desc(Bookshelf.date_started), desc(Bookshelf.updated_at))
        )
        return self.db.execute(q).scalars().all()

    def get_stats(self, *, user_id: str) -> Dict[str, Any]:
        items = self.db.execute(
            select(Bookshelf)
            .where(Bookshelf.user_id == user_id)
            .where(Bookshelf.shelf_status == "read")
            .where(Bookshelf.date_finished.isnot(None))
        ).scalars().all()

        now = _now()
        this_year = now.year
        this_month = (now.year, now.month)

        read_year = 0
        read_month = 0

        durations_days: List[float] = []
        finished_dates = []

        for it in items:
            df = it.date_finished
            if df is None:
                continue

            if df.year == this_year:
                read_year += 1
            if (df.year, df.month) == this_month:
                read_month += 1

            start = it.date_started or it.date_added
            if start and df >= start:
                durations_days.append((df - start).total_seconds() / 86400.0)

            finished_dates.append(df.date())

        avg_days = (sum(durations_days) / len(durations_days)) if durations_days else None

        unique_days = sorted(set(finished_dates))
        best = 0
        streak = 0
        prev = None
        for d in unique_days:
            if prev is None:
                streak = 1
            else:
                streak = streak + 1 if (d - prev).days == 1 else 1
            best = max(best, streak)
            prev = d

        current = 0
        if unique_days:
            last = unique_days[-1]
            gap = (now.date() - last).days
            if gap in (0, 1):
                current = 1
                for i in range(len(unique_days) - 2, -1, -1):
                    if (unique_days[i + 1] - unique_days[i]).days == 1:
                        current += 1
                    else:
                        break

        return {
            "read_this_month": read_month,
            "read_this_year": read_year,
            "avg_days_to_finish": avg_days,
            "current_streak_days": current,
            "best_streak_days": best,
        }
