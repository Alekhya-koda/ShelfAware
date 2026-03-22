#!/usr/bin/env python3
"""Test the /books/ endpoint to see what book_id type is returned."""

import requests
import json

endpoint = "http://localhost:8000/books/"

print("=" * 60)
print("Testing /books/ endpoint")
print("=" * 60)

try:
    response = requests.get(endpoint)
    response.raise_for_status()
    
    books = response.json()
    print(f"Status: {response.status_code}")
    print(f"Number of books: {len(books)}")
    
    if books:
        first_book = books[0]
        print(f"\nFirst book:")
        print(json.dumps(first_book, indent=2))
        print(f"\nbook_id value: {first_book.get('book_id')}")
        print(f"book_id type: {type(first_book.get('book_id'))}")
        print(f"book_id is string?: {isinstance(first_book.get('book_id'), str)}")
        
except Exception as e:
    print(f"Error: {e}")
