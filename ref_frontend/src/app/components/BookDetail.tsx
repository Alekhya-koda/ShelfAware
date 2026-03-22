import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Textarea } from './ui/textarea';
import { Separator } from './ui/separator';
import { Avatar, AvatarFallback } from './ui/avatar';
import { Star, ArrowLeft, Loader2 } from 'lucide-react';
import { apiService, Book, BookshelfItem, Review, ReviewCreate } from '../services/api';
import { toast } from 'sonner';

interface BookDetailProps {
  accessToken: string | null;
}

const readingMoods = [
  'Curious',
  'Inspired',
  'Excited',
  'Reflective',
  'Suspenseful',
  'Peaceful',
  'Moved',
  'Confused',
  'Hopeful',
];

function parseReadingCheckIn(synopsis?: string | null): { progress: number; moods: string[] } {
  if (!synopsis) return { progress: 0, moods: [] };
  try {
    const parsed = JSON.parse(synopsis) as { progress_percent?: number; mood?: string; moods?: string[] };
    const parsedMoods = Array.isArray(parsed.moods)
      ? parsed.moods.map((m) => String(m).trim()).filter(Boolean)
      : (typeof parsed.mood === 'string'
          ? parsed.mood.split(',').map((m) => m.trim()).filter(Boolean)
          : []);
    return {
      progress: typeof parsed.progress_percent === 'number' ? Math.max(0, Math.min(100, parsed.progress_percent)) : 0,
      moods: parsedMoods,
    };
  } catch {
    return { progress: 0, moods: [] };
  }
}

export function BookDetail({ accessToken }: BookDetailProps) {
  const { bookId } = useParams();
  const navigate = useNavigate();
  const [book, setBook] = useState<Book | null>(null);
  const [reviews, setReviews] = useState<Review[]>([]);
  const [myShelfItem, setMyShelfItem] = useState<BookshelfItem | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [myRating, setMyRating] = useState(0);
  const [reviewText, setReviewText] = useState('');
  const [submittingReview, setSubmittingReview] = useState(false);
  const [readingProgress, setReadingProgress] = useState(0);
  const [selectedMoods, setSelectedMoods] = useState<string[]>([]);
  const [savingCheckIn, setSavingCheckIn] = useState(false);

  useEffect(() => {
    const fetchBookData = async () => {
      if (!bookId) return;

      try {
        setLoading(true);
        const [bookData, reviewsData] = await Promise.all([
          apiService.getBook(bookId),
          apiService.getReviewsForBook(bookId)
        ]);

        let shelfItem: BookshelfItem | null = null;
        if (accessToken) {
          const myShelf = await apiService.getMyBookshelf(accessToken);
          shelfItem = myShelf.find((item) => item.book_id === bookId) ?? null;
        }

        setBook(bookData);
        setReviews(reviewsData);
        setMyShelfItem(shelfItem);
        const checkIn = parseReadingCheckIn(shelfItem?.synopsis);
        setReadingProgress(checkIn.progress);
        setSelectedMoods(checkIn.moods);
        setError(null);
      } catch (err) {
        setError('Failed to load book data');
        console.error('Error fetching book data:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchBookData();
  }, [bookId, accessToken]);

  const canReview = myShelfItem?.shelf_status === 'read';
  const isCurrentlyReading = myShelfItem?.shelf_status === 'currently_reading';

  const toggleMood = (mood: string) => {
    setSelectedMoods((prev) =>
      prev.includes(mood) ? prev.filter((m) => m !== mood) : [...prev, mood]
    );
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <Loader2 className="h-8 w-8 animate-spin" />
        <span className="ml-2">Loading book...</span>
      </div>
    );
  }

  if (error || !book) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <p className="text-red-500 mb-4">{error || 'Book not found'}</p>
          <div className="flex items-center justify-center gap-3">
            {accessToken && (
              <Button onClick={() => navigate('/bookshelf')}>
                <ArrowLeft className="size-4 mr-2" />
                Back to Bookshelf
              </Button>
            )}
            <Button variant={accessToken ? 'secondary' : 'default'} onClick={() => navigate('/inspiration')}>
              Back to Inspiration
            </Button>
          </div>
        </div>
      </div>
    );
  }

  const handleSubmitReview = async () => {
    if (!bookId || myRating === 0 || !accessToken) {
      return;
    }

    const payload: ReviewCreate = {
      rating: myRating,
      comment: reviewText.trim() || undefined,
      mood: selectedMoods.length > 0 ? selectedMoods.join(', ') : undefined,
    };

    try {
      setSubmittingReview(true);
      const createdReview = await apiService.addReview(accessToken, bookId, payload);
      setReviews((prevReviews) => [createdReview, ...prevReviews]);
      setMyRating(0);
      setReviewText('');
      toast.success('Review submitted successfully');
    } catch (submitErr) {
      console.error('Error submitting review:', submitErr);
      toast.error('Failed to submit review. Please try again.');
    } finally {
      setSubmittingReview(false);
    }
  };

  const handleSaveCheckIn = async () => {
    if (!bookId || !accessToken || !isCurrentlyReading) return;

    try {
      setSavingCheckIn(true);
      const updated = await apiService.updateBookshelfProgress(accessToken, bookId, {
        progress_percent: readingProgress,
        moods: selectedMoods,
      });
      setMyShelfItem(updated);
      toast.success('Reading check-in saved');
    } catch (err) {
      console.error('Error saving reading check-in:', err);
      toast.error(err instanceof Error ? err.message : 'Failed to save reading progress');
    } finally {
      setSavingCheckIn(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3 mb-4">
        {accessToken && (
          <Button onClick={() => navigate('/bookshelf')}>
            <ArrowLeft className="size-4 mr-2" />
            Back to Bookshelf
          </Button>
        )}
        <Button variant={accessToken ? 'secondary' : 'ghost'} onClick={() => navigate('/inspiration')}>
          Back to Inspiration
        </Button>
      </div>

      <div className="grid md:grid-cols-3 gap-6">
        {/* Book Cover and Info */}
        <div className="md:col-span-1">
          <Card>
            <CardContent className="p-6">
              <img
                src={book.cover_image_url || 'https://via.placeholder.com/400x600?text=No+Cover'}
                alt={book.title}
                className="w-full rounded-lg shadow-lg mb-4"
              />
              <h1 className="text-2xl font-bold mb-2">{book.title}</h1>
              {book.subtitle && (
                <p className="text-gray-600 mb-4">{book.subtitle}</p>
              )}

              {book.abstract && (
                <p className="text-sm text-gray-700 mb-4">{book.abstract}</p>
              )}

              <Separator className="my-4" />

              <div className="space-y-2">
                {book.page_count && (
                  <p className="text-sm text-gray-600">
                    <strong>Pages:</strong> {book.page_count}
                  </p>
                )}
                {book.published_date && (
                  <p className="text-sm text-gray-600">
                    <strong>Published:</strong> {new Date(book.published_date).getFullYear()}
                  </p>
                )}
                {book.CommunitySynopsis && (
                  <div className="mt-4">
                    <h3 className="text-sm font-semibold mb-2">Community Synopsis</h3>
                    <p className="text-sm text-gray-700">{book.CommunitySynopsis}</p>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Main Content */}
        <div className="md:col-span-2 space-y-6">

          {/* About the Book */}
          <Card>
            <CardHeader>
              <CardTitle>About this Book</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-gray-700">{book.abstract}</p>
            </CardContent>
          </Card>

          {/* Reading Check-In (Only for currently reading) */}
          {isCurrentlyReading && (
            <Card>
              <CardHeader>
                <CardTitle>Reading Check-In</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">Progress: {readingProgress}%</label>
                  <input
                    type="range"
                    min={0}
                    max={100}
                    step={1}
                    value={readingProgress}
                    onChange={(e) => setReadingProgress(Number(e.target.value))}
                    className="w-full"
                    disabled={savingCheckIn}
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">How does this book make you feel so far?</label>
                  <div className="flex flex-wrap gap-2">
                    {readingMoods.map((m) => (
                      <Badge
                        key={m}
                        variant={selectedMoods.includes(m) ? 'default' : 'outline'}
                        className={`cursor-pointer ${savingCheckIn ? 'opacity-50 pointer-events-none' : ''}`}
                        onClick={() => toggleMood(m)}
                      >
                        {m}
                      </Badge>
                    ))}
                  </div>
                </div>

                <Button onClick={handleSaveCheckIn} className="w-full" disabled={savingCheckIn}>
                  {savingCheckIn ? (
                    <>
                      <Loader2 className="size-4 mr-2 animate-spin" />
                      Saving...
                    </>
                  ) : (
                    'Save Check-In'
                  )}
                </Button>
              </CardContent>
            </Card>
          )}

          {/* Write Review (Only when completed) */}
          {canReview ? (
            <Card>
              <CardHeader>
                <CardTitle>Write a Review</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">Your Rating</label>
                  <div className="flex gap-2">
                    {[1, 2, 3, 4, 5].map((star) => (
                      <button
                        key={star}
                        onClick={() => setMyRating(star)}
                        className="focus:outline-none"
                        disabled={submittingReview}
                      >
                        <Star
                          className={`size-8 ${
                            star <= myRating
                              ? 'text-yellow-500 fill-current'
                              : 'text-gray-300'
                          }`}
                        />
                      </button>
                    ))}
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">How did this book make you feel?</label>
                  <div className="flex flex-wrap gap-2">
                    {readingMoods.map((m) => (
                      <Badge
                        key={m}
                        variant={selectedMoods.includes(m) ? 'default' : 'outline'}
                        className={`cursor-pointer ${submittingReview ? 'opacity-50 pointer-events-none' : ''}`}
                        onClick={() => toggleMood(m)}
                      >
                        {m}
                      </Badge>
                    ))}
                  </div>
                  <p className="text-xs text-gray-500 mt-2">
                    Selected moods from reading check-ins are synced here and can still be adjusted before submitting.
                  </p>
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">Your Review (Optional)</label>
                  <Textarea
                    placeholder="Share your thoughts about this book..."
                    value={reviewText}
                    onChange={(e) => setReviewText(e.target.value)}
                    rows={4}
                    disabled={submittingReview}
                  />
                </div>

                <Button
                  onClick={handleSubmitReview}
                  className="w-full"
                  disabled={submittingReview || myRating === 0}
                >
                  {submittingReview ? (
                    <>
                      <Loader2 className="size-4 mr-2 animate-spin" />
                      Submitting...
                    </>
                  ) : (
                    'Submit Review'
                  )}
                </Button>
              </CardContent>
            </Card>
          ) : (
            <Card>
              <CardHeader>
                <CardTitle>Write a Review</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-gray-600">
                  You can submit a review after marking this book as read.
                </p>
              </CardContent>
            </Card>
          )}

          {/* Existing Reviews */}
          <Card>
            <CardHeader>
              <CardTitle>Reviews ({reviews.length})</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {reviews.length === 0 ? (
                <p className="text-gray-500 text-center py-4">No reviews yet. Be the first to review this book!</p>
              ) : (
                reviews.map((review) => (
                  <div key={review.review_id} className="border-b pb-4 last:border-b-0">
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center">
                        <Avatar className="size-10 mr-3">
                          <AvatarFallback>U{review.user_id.slice(-1).toUpperCase()}</AvatarFallback>
                        </Avatar>
                        <div>
                          <div className="flex items-center">
                            {Array.from({ length: 5 }).map((_, i) => (
                              <Star
                                key={i}
                                className={`size-4 ${
                                  i < review.rating
                                    ? 'text-yellow-500 fill-current'
                                    : 'text-gray-300'
                                }`}
                              />
                            ))}
                            <span className="ml-2 text-sm text-gray-600">
                              {new Date(review.created_at).toLocaleDateString()}
                            </span>
                          </div>
                          {review.mood && (
                            <p className="text-sm text-gray-500 mt-1">Feeling: {review.mood}</p>
                          )}
                        </div>
                      </div>
                    </div>
                    {(review.comment || review.body) && (
                      <p className="text-gray-700 mt-2">{review.comment || review.body}</p>
                    )}
                  </div>
                ))
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}