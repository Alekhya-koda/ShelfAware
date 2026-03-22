import React, { useEffect, useMemo, useState } from 'react';
import { Card, CardContent } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Input } from './ui/input';
import { BookOpen, Loader2, Search, Trash2 } from 'lucide-react';
import { apiService, Book, BookshelfItem, BookshelfStats, ShelfStatus } from '../services/api';
import { useNavigate } from 'react-router';
import { toast } from 'sonner';

interface BookshelfProps {
  accessToken: string | null;
}

type ShelfBook = BookshelfItem & {
  book?: Book;
};

export function Bookshelf({ accessToken }: BookshelfProps) {
  const navigate = useNavigate();
  const [books, setBooks] = useState<Book[]>([]);
  const [shelfItems, setShelfItems] = useState<BookshelfItem[]>([]);
  const [stats, setStats] = useState<BookshelfStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [savingBookId, setSavingBookId] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      if (!accessToken) {
        setLoading(false);
        setError('Please sign in to view your bookshelf.');
        return;
      }

      try {
        setLoading(true);
        const [fetchedBooks, fetchedShelf] = await Promise.all([
          apiService.getBooks(),
          apiService.getMyBookshelf(accessToken),
        ]);
        const fetchedStats = await apiService.getMyBookshelfStats(accessToken).catch(() => null);
        setBooks(fetchedBooks);
        setShelfItems(fetchedShelf);
        setStats(fetchedStats);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load bookshelf.');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [accessToken]);

  const shelfBooks = useMemo<ShelfBook[]>(() => {
    const map = new Map(books.map((b) => [b.book_id, b]));
    return shelfItems.map((item) => ({ ...item, book: map.get(item.book_id) }));
  }, [books, shelfItems]);

  const filterByStatus = (status: ShelfStatus) => {
    return shelfBooks.filter((item) => {
      const title = item.book?.title || item.book_id;
      const matchesSearch = title.toLowerCase().includes(searchQuery.toLowerCase());
      return item.shelf_status === status && matchesSearch;
    });
  };

  const wantToReadBooks = filterByStatus('want_to_read');
  const currentlyReadingBooks = filterByStatus('currently_reading');
  const completedBooks = filterByStatus('read');

  const handleStatusChange = async (bookId: string, next: ShelfStatus) => {
    if (!accessToken) return;
    try {
      setSavingBookId(bookId);
      const updated = await apiService.updateBookshelfStatus(accessToken, bookId, next);
      setShelfItems((prev) => prev.map((item) => (item.book_id === updated.book_id ? updated : item)));
      toast.success('Bookshelf updated');
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to update shelf status');
    } finally {
      setSavingBookId(null);
    }
  };

  const handleRemove = async (bookId: string) => {
    if (!accessToken) return;
    try {
      setSavingBookId(bookId);
      await apiService.removeFromBookshelf(accessToken, bookId);
      setShelfItems((prev) => prev.filter((item) => item.book_id !== bookId));
      toast.success('Removed from bookshelf');
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to remove book');
    } finally {
      setSavingBookId(null);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <Loader2 className="h-8 w-8 animate-spin" />
        <span className="ml-2">Loading your bookshelf...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <p className="text-red-500 mb-4">{error}</p>
          <Button onClick={() => window.location.reload()}>Reload</Button>
        </div>
      </div>
    );
  }

  const BookCard = ({ item }: { item: ShelfBook }) => (
    <Card
      className="cursor-pointer hover:shadow-lg transition-shadow h-full flex flex-col"
      onClick={() => navigate(`/book/${item.book_id}`)}
    >
      <div className="aspect-[2/3] relative overflow-hidden rounded-t-lg">
        <img
          src={item.book?.cover_image_url || 'https://via.placeholder.com/400x600?text=No+Cover'}
          alt={item.book?.title || item.book_id}
          className="w-full h-full object-cover"
        />
      </div>
      <CardContent className="p-3 flex-1 flex flex-col">
        <h3 className="font-semibold line-clamp-2 text-sm mb-2 flex-grow">{item.book?.title || item.book_id}</h3>
        <div className="flex gap-1" onClick={(e) => e.stopPropagation()}>
          {item.shelf_status === 'want_to_read' && (
            <Button size="sm" variant="outline" onClick={() => handleStatusChange(item.book_id, 'currently_reading')} disabled={savingBookId === item.book_id} className="text-xs h-7 px-2">
              Start
            </Button>
          )}
          {item.shelf_status === 'currently_reading' && (
            <Button size="sm" variant="outline" onClick={() => handleStatusChange(item.book_id, 'read')} disabled={savingBookId === item.book_id} className="text-xs h-7 px-2">
              Finish
            </Button>
          )}
          <Button size="sm" variant="ghost" onClick={() => handleRemove(item.book_id)} disabled={savingBookId === item.book_id} className="ml-auto h-7 px-1">
            <Trash2 className="size-3" />
          </Button>
        </div>
      </CardContent>
    </Card>
  );

  const ShelfSection = ({ title, items }: { title: string; items: ShelfBook[] }) => (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-900">{title}</h2>
        <Badge variant="secondary" className="text-lg px-3 py-1">
          {items.length}
        </Badge>
      </div>
      {items.length === 0 ? (
        <div className="bg-gray-50 rounded-lg p-8 text-center text-gray-500">
          <p>No books in this section yet.</p>
        </div>
      ) : (
        <div className="overflow-x-auto pb-4">
          <div className="flex gap-4 min-w-min">
            {items.map((item) => (
              <div key={item.book_id} className="flex-shrink-0 w-[160px]">
                <BookCard item={item} />
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h1 className="text-3xl font-bold mb-2">Bookshelf</h1>
        <p className="text-gray-600">Your personal shelf synced to your account</p>
      </div>

      <Card>
        <CardContent className="p-4">
          <div className="relative max-w-md mx-auto">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 size-4" />
            <Input
              placeholder="Search your bookshelf..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10"
            />
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardContent className="p-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
            <div className="rounded-md bg-gray-50 p-3">
              <p className="text-gray-500">Read This Month</p>
              <p className="text-xl font-semibold">{stats?.read_this_month ?? 0}</p>
            </div>
            <div className="rounded-md bg-gray-50 p-3">
              <p className="text-gray-500">Read This Year</p>
              <p className="text-xl font-semibold">{stats?.read_this_year ?? 0}</p>
            </div>
            <div className="rounded-md bg-gray-50 p-3">
              <p className="text-gray-500">Avg Days to Finish</p>
              <p className="text-xl font-semibold">
                {typeof stats?.avg_days_to_finish === 'number' ? stats.avg_days_to_finish.toFixed(1) : '-'}
              </p>
            </div>
            <div className="rounded-md bg-gray-50 p-3">
              <p className="text-gray-500">Current Streak</p>
              <p className="text-xl font-semibold">{stats?.current_streak_days ?? 0}</p>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="space-y-12">
        <ShelfSection title="Want to Read" items={wantToReadBooks} />
        <ShelfSection title="Currently Reading" items={currentlyReadingBooks} />
        <ShelfSection title="Finished Reading" items={completedBooks} />
      </div>

      {shelfItems.length === 0 && (
        <div className="text-center py-12">
          <BookOpen className="mx-auto h-12 w-12 text-gray-400 mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">Your shelf is empty</h3>
          <p className="text-gray-500 mb-4">Browse Inspiration and add books to start tracking your reading.</p>
          <Button onClick={() => navigate('/inspiration')}>Go to Inspiration</Button>
        </div>
      )}
    </div>
  );
}
