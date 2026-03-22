#!/usr/bin/env python3
"""Test BookshelfCreate schema validation directly."""

from pydantic import BaseModel
from typing import Literal

ShelfStatus = Literal["want_to_read", "currently_reading", "read"]

class BookshelfCreate(BaseModel):
    book_id: str

# Test payloads
test_payloads = [
    {"book_id": "1842"},
    {"book_id": "6462"},
    {"book_id": 1842},  # integer instead of string
    {},  # missing book_id
]

print("=" * 60)
print("Testing BookshelfCreate Validation")
print("=" * 60)

for i, payload in enumerate(test_payloads, 1):
    print(f"\nTest {i}: {payload}")
    try:
        result = BookshelfCreate(**payload)
        print(f"✓ Success: {result}")
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {e}")
