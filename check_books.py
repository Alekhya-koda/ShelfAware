#!/usr/bin/env python3
"""Check if book 1842 exists in the database."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.db.database import SessionLocal
from app.models.book import Book
from sqlalchemy import select

db = SessionLocal()

# Check if book 1842 exists
print("=" * 60)
print("Checking if book '1842' exists in database...")
print("=" * 60)

result = db.execute(select(Book).where(Book.book_id == "1842")).scalar_one_or_none()

if result:
    print(f"\n✓ Book found!")
    print(f"  ID: {result.book_id}")
    print(f"  Title: {result.title}")
    print(f"  Type of book_id: {type(result.book_id)}")
else:
    print(f"\n✗ Book '1842' NOT found in database")

# List first 5 books to see what IDs we have
print("\n" + "=" * 60)
print("First 10 books in database:")
print("=" * 60)

books = db.execute(select(Book).limit(10)).scalars().all()
for i, book in enumerate(books, 1):
    print(f"{i}. ID: {book.book_id} (type: {type(book.book_id).__name__}) - {book.title}")

db.close()
