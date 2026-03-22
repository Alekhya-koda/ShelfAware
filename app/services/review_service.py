# app/services/review_service.py

from __future__ import annotations
from typing import Sequence, Optional

from fastapi import HTTPException, status
from sqlalchemy import select, func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.review import Review
from app.models.book import Book
from app.models.user import User

from app.schemas.review import ReviewCreate, ReviewUpdate


class ReviewService:
    def __init__(self, db: Session):
        self.db = db

    # --- Internal Helpers ---
    def _ensure_book_exists(self, book_id: str) -> str:
        title = self.db.scalar(select(Book.title).where(Book.book_id == book_id))
        if not title:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Book not found")
        return str(title)

    def _ensure_user_exists(self, user_id: str) -> None:
        exists = self.db.scalar(select(User.user_id).where(User.user_id == user_id))
        if not exists:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    def _get_review_or_404(self, review_id: str) -> Review:
        review = self.db.get(Review, review_id)
        if not review:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Review not found")
        return review

    # --- Commands ---
    def add_review(self, *, book_id: str, user_id: str, review_data: ReviewCreate) -> Review:
        book_title = self._ensure_book_exists(book_id)
        self._ensure_user_exists(user_id)

        payload = review_data.model_dump(exclude_unset=True)

        # Pull API-only fields (not DB columns)
        book_mood_text: Optional[str] = payload.pop("book_mood", None)
        legacy_mood_text: Optional[str] = payload.pop("mood", None)
        comment_text: Optional[str] = payload.pop("comment", None)
        if book_mood_text is None:
            book_mood_text = legacy_mood_text

        # Map API comment -> DB body if provided
        if comment_text is not None and payload.get("body") is None:
            payload["body"] = comment_text

        # Persist the book title on the review row for downstream UI consumers.
        if not payload.get("title"):
            payload["title"] = book_title

        rating = payload.get("rating")
        if rating is None or not (1 <= rating <= 5):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Rating must be between 1 and 5",
            )

        review = Review(book_id=book_id, user_id=user_id, **payload)
        # Book-level mood is attached to the review response, not daily user mood logs.
        setattr(review, "book_mood", (book_mood_text or "").strip() or None)
        setattr(review, "mood", getattr(review, "book_mood", None))
        self.db.add(review)

        try:
            self.db.commit()
            self.db.refresh(review)

            # optional: attach comment for response serialization convenience
            # (does not persist to DB, just helps schemas expecting "comment")
            setattr(review, "comment", review.body)
            setattr(review, "book_mood", (book_mood_text or "").strip() or None)
            setattr(review, "mood", getattr(review, "book_mood", None))

            return review
        except IntegrityError as e:
            self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="You have already reviewed this book.",
            ) from e

    def update_review(self, review_id: str, acting_user_id: str, review_data: ReviewUpdate) -> Review:
        self._ensure_user_exists(acting_user_id)

        review = self._get_review_or_404(review_id)
        if review.user_id != acting_user_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to edit this review")

        update_data = review_data.model_dump(exclude_unset=True)

        book_mood_text: Optional[str] = update_data.pop("book_mood", None)
        legacy_mood_text: Optional[str] = update_data.pop("mood", None)
        comment_text: Optional[str] = update_data.pop("comment", None)
        if book_mood_text is None:
            book_mood_text = legacy_mood_text

        # Map API comment -> DB body if provided
        if comment_text is not None and "body" not in update_data:
            update_data["body"] = comment_text

        # Validate rating if provided
        if "rating" in update_data and update_data["rating"] is not None:
            rating = update_data["rating"]
            if not (1 <= rating <= 5):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Rating must be between 1 and 5",
                )

        # Update DB fields
        for key, value in update_data.items():
            setattr(review, key, value)

        self.db.commit()
        self.db.refresh(review)

        setattr(review, "comment", review.body)
        if book_mood_text is not None:
            setattr(review, "book_mood", (book_mood_text or "").strip() or None)
            setattr(review, "mood", getattr(review, "book_mood", None))
        return review

    def delete_review(self, review_id: str, acting_user_id: str) -> None:
        self._ensure_user_exists(acting_user_id)

        review = self._get_review_or_404(review_id)
        if review.user_id != acting_user_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to delete this review")

        self.db.delete(review)
        self.db.commit()

    # --- Queries ---
    def get_reviews_by_book_id(
        self,
        book_id: str,
        limit: int = 20,
        offset: int = 0,
        newest_first: bool = True,
    ) -> Sequence[Review]:
        self._ensure_book_exists(book_id)

        stmt = (
            select(Review)
            .where(Review.book_id == book_id)
            .order_by(Review.created_at.desc() if newest_first else Review.created_at.asc())
            .limit(limit)
            .offset(offset)
        )

        reviews = self.db.scalars(stmt).all()
        for r in reviews:
            setattr(r, "comment", r.body)
            if not hasattr(r, "book_mood"):
                setattr(r, "book_mood", None)
            if not hasattr(r, "mood"):
                setattr(r, "mood", getattr(r, "book_mood", None))
        return reviews

    def get_average_rating(self, book_id: str) -> float | None:
        self._ensure_book_exists(book_id)
        avg = self.db.scalar(select(func.avg(Review.rating)).where(Review.book_id == book_id))
        return round(float(avg), 2) if avg is not None else None
