import React, { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './ui/table';
import { BookOpen, Sparkles, CheckCircle, AlertCircle, Edit2, Trash2, Plus, X } from 'lucide-react';
import { apiService, Book, SynopsisModerationItem } from '../services/api';
import { toast } from 'sonner';

interface BookFormData {
  title: string;
  subtitle: string;
  cover_image_url: string;
  abstract: string;
  page_count: number | null;
  published_date: string;
}

interface AdminPanelProps {
  accessToken: string | null;
}

const initialFormData: BookFormData = {
  title: '',
  subtitle: '',
  cover_image_url: '',
  abstract: '',
  page_count: null,
  published_date: '',
};

export function AdminPanel({ accessToken }: AdminPanelProps) {
  const [books, setBooks] = useState<Book[]>([]);
  const [isLoadingBooks, setIsLoadingBooks] = useState(true);
  const [isSyncingSynopsis, setIsSyncingSynopsis] = useState(false);
  const [isLoadingModeration, setIsLoadingModeration] = useState(false);
  const [moderationItems, setModerationItems] = useState<SynopsisModerationItem[]>([]);
  const [processingModerationId, setProcessingModerationId] = useState<string | null>(null);
  
  // Book management states
  const [showAddBookForm, setShowAddBookForm] = useState(false);
  const [isSubmittingBook, setIsSubmittingBook] = useState(false);
  const [formData, setFormData] = useState<BookFormData>(initialFormData);
  const [editingBookId, setEditingBookId] = useState<string | null>(null);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState<string | null>(null);

  const loadBooks = async () => {
    try {
      setIsLoadingBooks(true);
      const dbBooks = await apiService.getBooks();
      setBooks(dbBooks);
    } catch (error) {
      console.error('Error loading books:', error);
      toast.error('Failed to load books from database');
    } finally {
      setIsLoadingBooks(false);
    }
  };

  useEffect(() => {
    loadBooks();
    loadModerationItems();
  }, []);

  const loadModerationItems = async () => {
    try {
      setIsLoadingModeration(true);
      const response = await apiService.getSynopsisModeration('pending');
      setModerationItems(response.items || []);
    } catch (error) {
      console.error('Error loading moderation items:', error);
      toast.error('Failed to load synopsis moderation queue');
    } finally {
      setIsLoadingModeration(false);
    }
  };

  const handleSyncSynopsis = async () => {
    setIsSyncingSynopsis(true);
    try {
      const result = await apiService.triggerCommunityReviewGeneration();
      await Promise.all([loadBooks(), loadModerationItems()]);
      toast.success(
        `Generation complete. Proposed: ${result.proposed}, Refreshed: ${result.refreshed}, Skipped: ${result.skipped}`
      );
    } catch (error) {
      console.error('Error syncing summaries:', error);
      toast.error('Failed to generate community reviews');
    } finally {
      setIsSyncingSynopsis(false);
    }
  };

  const handleModerationAction = async (moderationId: string, action: 'accept' | 'reject') => {
    setProcessingModerationId(moderationId);
    try {
      if (action === 'accept') {
        await apiService.acceptSynopsisModeration(moderationId);
        toast.success('Community review accepted and updated');
      } else {
        await apiService.rejectSynopsisModeration(moderationId);
        toast.success('Community review proposal rejected');
      }
      await Promise.all([loadBooks(), loadModerationItems()]);
    } catch (error) {
      console.error('Error handling moderation action:', error);
      toast.error('Failed to process moderation action');
    } finally {
      setProcessingModerationId(null);
    }
  };

  const handleFormChange = (field: keyof BookFormData, value: any) => {
    setFormData((prev) => ({
      ...prev,
      [field]: value,
    }));
  };

  const resetForm = () => {
    setFormData(initialFormData);
    setEditingBookId(null);
    setShowAddBookForm(false);
  };

  const startEditBook = (book: Book) => {
    setEditingBookId(book.book_id);
    setFormData({
      title: book.title,
      subtitle: book.subtitle || '',
      cover_image_url: book.cover_image_url || '',
      abstract: book.abstract || '',
      page_count: book.page_count || null,
      published_date: book.published_date || '',
    });
    setShowAddBookForm(true);
  };

  const handleSubmitBook = async () => {
    if (!accessToken) {
      toast.error('Not authenticated');
      return;
    }

    if (!formData.title.trim()) {
      toast.error('Book title is required');
      return;
    }

    setIsSubmittingBook(true);
    try {
      if (editingBookId) {
        // Update existing book
        await apiService.updateBook(accessToken, editingBookId, formData);
        toast.success('Book updated successfully');
      } else {
        // Create new book
        await apiService.createBook(accessToken, formData);
        toast.success('Book added successfully');
      }
      await loadBooks();
      resetForm();
    } catch (error) {
      console.error('Error saving book:', error);
      toast.error('Failed to save book');
    } finally {
      setIsSubmittingBook(false);
    }
  };

  const handleDeleteBook = async (bookId: string) => {
    if (!accessToken) {
      toast.error('Not authenticated');
      return;
    }

    setIsSubmittingBook(true);
    try {
      await apiService.deleteBook(accessToken, bookId);
      toast.success('Book deleted successfully');
      await loadBooks();
      setShowDeleteConfirm(null);
    } catch (error) {
      console.error('Error deleting book:', error);
      toast.error('Failed to delete book');
    } finally {
      setIsSubmittingBook(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Admin Panel</h1>
          <p className="text-gray-600">Manage book data and moderate community review updates</p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={loadBooks} disabled={isLoadingBooks}>
            Refresh Books
          </Button>
          <Button onClick={handleSyncSynopsis} disabled={isSyncingSynopsis}>
            <Sparkles className="size-4 mr-2" />
            {isSyncingSynopsis ? 'Generating...' : 'Generate Community Reviews'}
          </Button>
        </div>
      </div>

      <Tabs defaultValue="books" className="w-full">
        <TabsList>
          <TabsTrigger value="books">
            <BookOpen className="size-4 mr-2" />
            Book Management ({books.length})
          </TabsTrigger>
          <TabsTrigger value="reviews">
            <AlertCircle className="size-4 mr-2" />
            Community Review Moderation ({moderationItems.length} pending)
          </TabsTrigger>
        </TabsList>

        <TabsContent value="books" className="mt-6 space-y-6">
          {/* Add Book Form */}
          {showAddBookForm && (
            <Card className="bg-blue-50 border-blue-200">
              <CardHeader className="flex flex-row items-center justify-between pb-3">
                <CardTitle>{editingBookId ? 'Edit Book' : 'Add New Book'}</CardTitle>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={resetForm}
                  className="h-8 w-8 p-0"
                >
                  <X className="h-4 w-4" />
                </Button>
              </CardHeader>
              <CardContent>
                <div className="grid md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Title *</label>
                    <input
                      type="text"
                      placeholder="Book title"
                      value={formData.title}
                      onChange={(e) => handleFormChange('title', e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Subtitle</label>
                    <input
                      type="text"
                      placeholder="Book subtitle"
                      value={formData.subtitle}
                      onChange={(e) => handleFormChange('subtitle', e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Cover Image URL</label>
                    <input
                      type="text"
                      placeholder="https://..."
                      value={formData.cover_image_url}
                      onChange={(e) => handleFormChange('cover_image_url', e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Published Date</label>
                    <input
                      type="date"
                      value={formData.published_date}
                      onChange={(e) => handleFormChange('published_date', e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Page Count</label>
                    <input
                      type="number"
                      placeholder="Number of pages"
                      value={formData.page_count || ''}
                      onChange={(e) => handleFormChange('page_count', e.target.value ? parseInt(e.target.value) : null)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                </div>
                <div className="mt-4">
                  <label className="block text-sm font-medium text-gray-700 mb-1">Abstract</label>
                  <textarea
                    placeholder="Book abstract or description"
                    value={formData.abstract}
                    onChange={(e) => handleFormChange('abstract', e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    rows={3}
                  />
                </div>
                <div className="mt-6 flex gap-2">
                  <Button
                    onClick={handleSubmitBook}
                    disabled={isSubmittingBook}
                    className="bg-blue-600 hover:bg-blue-700"
                  >
                    {isSubmittingBook ? 'Saving...' : editingBookId ? 'Update Book' : 'Add Book'}
                  </Button>
                  <Button variant="outline" onClick={resetForm}>
                    Cancel
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}

          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle>Book Catalog</CardTitle>
              {!showAddBookForm && (
                <Button onClick={() => setShowAddBookForm(true)} className="gap-2">
                  <Plus className="h-4 w-4" />
                  Add Book
                </Button>
              )}
            </CardHeader>
            <CardContent>
              <Table className="w-full">
                <TableHeader>
                  <TableRow>
                    <TableHead>Cover</TableHead>
                    <TableHead className="w-[200px]">Title</TableHead>
                    <TableHead className="w-[280px]">Community Review</TableHead>
                    <TableHead>Pages</TableHead>
                    <TableHead>Published</TableHead>
                    <TableHead className="w-[120px]">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {isLoadingBooks ? (
                    <TableRow>
                      <TableCell colSpan={6} className="text-center py-8 text-gray-500">
                        Loading books from app.db...
                      </TableCell>
                    </TableRow>
                  ) : books.length === 0 ? (
                    <TableRow>
                      <TableCell colSpan={6} className="text-center py-8 text-gray-500">
                        No books found in database
                      </TableCell>
                    </TableRow>
                  ) : (
                    books.map((book) => (
                      <TableRow key={book.book_id}>
                        <TableCell>
                          <img
                            src={book.cover_image_url || 'https://via.placeholder.com/80x120?text=No+Cover'}
                            alt={book.title}
                            className="w-12 h-16 object-cover rounded"
                          />
                        </TableCell>
                        <TableCell className="w-[200px] max-w-[200px]">
                          <span
                            className="font-medium whitespace-normal break-words leading-5"
                            style={{
                              display: '-webkit-box',
                              WebkitLineClamp: 2,
                              WebkitBoxOrient: 'vertical',
                              overflow: 'hidden',
                            }}
                            title={book.title}
                          >
                            {book.title}
                          </span>
                        </TableCell>
                        <TableCell className="w-[280px] max-w-[280px]">
                          <span
                            className="text-sm text-gray-700 whitespace-normal break-words leading-5"
                            style={{
                              display: '-webkit-box',
                              WebkitLineClamp: 3,
                              WebkitBoxOrient: 'vertical',
                              overflow: 'hidden',
                            }}
                            title={book.CommunitySynopsis || 'No community review available'}
                          >
                            {book.CommunitySynopsis || '-'}
                          </span>
                        </TableCell>
                        <TableCell>{book.page_count ?? '-'}</TableCell>
                        <TableCell>
                          {book.published_date ? new Date(book.published_date).getFullYear() : '-'}
                        </TableCell>
                        <TableCell>
                          <div className="flex gap-2">
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() => startEditBook(book)}
                              disabled={isSubmittingBook}
                              className="h-8 w-8 p-0"
                              title="Edit book"
                            >
                              <Edit2 className="h-4 w-4" />
                            </Button>
                            {showDeleteConfirm === book.book_id ? (
                              <>
                                <Button
                                  size="sm"
                                  variant="destructive"
                                  onClick={() => handleDeleteBook(book.book_id)}
                                  disabled={isSubmittingBook}
                                  className="h-8 px-2 text-xs"
                                >
                                  Confirm
                                </Button>
                                <Button
                                  size="sm"
                                  variant="outline"
                                  onClick={() => setShowDeleteConfirm(null)}
                                  disabled={isSubmittingBook}
                                  className="h-8 px-2 text-xs"
                                >
                                  Cancel
                                </Button>
                              </>
                            ) : (
                              <Button
                                size="sm"
                                variant="outline"
                                onClick={() => setShowDeleteConfirm(book.book_id)}
                                disabled={isSubmittingBook}
                                className="h-8 w-8 p-0 text-red-600 hover:text-red-700"
                                title="Delete book"
                              >
                                <Trash2 className="h-4 w-4" />
                              </Button>
                            )}
                          </div>
                        </TableCell>
                      </TableRow>
                    ))
                  )}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="reviews" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <AlertCircle className="size-5 mr-2 text-orange-500" />
                Pending Community Review Changes ({moderationItems.length})
              </CardTitle>
            </CardHeader>
            <CardContent>
              {isLoadingModeration ? (
                <p className="text-center text-gray-500 py-8">Loading moderation queue...</p>
              ) : moderationItems.length === 0 ? (
                <p className="text-center text-gray-500 py-8">No pending community review changes</p>
              ) : (
                <div className="space-y-4">
                  {moderationItems.map((item) => (
                    <div key={item.moderation_id} className="border rounded-lg p-4">
                      <div className="flex items-start justify-between mb-3">
                        <div>
                          <h4 className="font-semibold">{item.book_title}</h4>
                          <p className="text-sm text-gray-600">
                            Source reviews: {item.user_synopsis_count}
                          </p>
                        </div>
                        <div className="flex gap-2">
                          <Button
                            size="sm"
                            onClick={() => handleModerationAction(item.moderation_id, 'accept')}
                            disabled={processingModerationId === item.moderation_id}
                          >
                            <CheckCircle className="size-4 mr-1" />
                            Accept
                          </Button>
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => handleModerationAction(item.moderation_id, 'reject')}
                            disabled={processingModerationId === item.moderation_id}
                          >
                            Reject
                          </Button>
                        </div>
                      </div>

                      <div className="grid md:grid-cols-2 gap-3">
                        <div className="rounded-md border p-3 bg-gray-50">
                          <p className="text-xs uppercase tracking-wide text-gray-500 mb-2">Current Community Review</p>
                          <p className="text-sm text-gray-700 whitespace-pre-wrap">{item.current_synopsis || '-'}</p>
                        </div>
                        <div className="rounded-md border p-3 bg-orange-50">
                          <p className="text-xs uppercase tracking-wide text-gray-500 mb-2">Proposed Update</p>
                          <p className="text-sm text-gray-800 whitespace-pre-wrap">{item.proposed_synopsis}</p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}