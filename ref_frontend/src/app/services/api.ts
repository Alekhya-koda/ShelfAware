// API service for connecting to the ShelfAware backend
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://54.254.9.171:8000';

export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = 'ApiError';
  }
}

export interface Book {
  book_id: string;
  title: string;
  subtitle?: string;
  cover_image_url?: string;
  abstract?: string;
  CommunitySynopsis?: string;
  page_count?: number;
  published_date?: string;
  created_at: string;
}

export interface Review {
  review_id: string;
  book_id: string;
  user_id?: string;
  rating: number;
  title?: string;
  comment?: string;
  book_mood?: string;
  // Backward compatibility with older API payloads.
  mood?: string;
  created_at: string;
  updated_at?: string;
}

export interface ReviewCreate {
  rating: number;
  comment?: string;
  book_mood?: string;
  // Backward compatibility with older API payloads.
  mood?: string;
}

export interface ReviewUpdate {
  rating?: number;
  comment?: string;
  book_mood?: string;
  mood?: string;
}

export interface ChatRequest {
  message: string;
  user_id?: string;
}

export interface ChatBookRecommendation {
  book_id?: string;
  id?: string;
  title: string;
  author?: string;
  similarity?: number;
  subtitle?: string | null;
  abstract?: string | null;
  cover_image_url?: string | null;
  genres?: string[];
}

export interface ChatResponse {
  response: string;
  mood: string;
  books: ChatBookRecommendation[];
  follow_up_questions: string[];
}

export interface LoginResponse {
  message: string;
  user: {
    user_id: string;
    cognito_sub: string;
    email: string;
    status: string;
    created_at: string;
  };
  tokens: {
    id_token?: string;
    access_token?: string;
    refresh_token?: string;
  };
}

export interface UserProfile {
  user_id: string;
  display_name: string;
  profile_photo_url?: string | null;
  bio?: string | null;
  location?: string | null;
  favorite_genres_json?: string | null;
}

export interface UserProfileUpdate {
  display_name?: string;
  profile_photo_url?: string | null;
  bio?: string | null;
  location?: string | null;
  favorite_genres_json?: string | null;
}

export interface ConfirmAccountResponse {
  message: string;
}

export type ShelfStatus = 'want_to_read' | 'currently_reading' | 'read';

export interface BookshelfItem {
  user_id: string;
  book_id: string;
  shelf_status: ShelfStatus;
  date_added: string;
  date_started?: string | null;
  date_finished?: string | null;
  updated_at: string;
  synopsis?: string | null;
}

export interface ReadingCheckIn {
  progress_percent: number;
  book_mood?: string;
  book_moods?: string[];
  mood?: string;
  moods?: string[];
}

export interface BookshelfStats {
  read_this_month: number;
  read_this_year: number;
  avg_days_to_finish?: number | null;
  current_streak_days: number;
  best_streak_days: number;
}

export interface SynopsisSyncResult {
  status: string;
  timestamp: string;
  total_books_processed: number;
  proposed: number;
  refreshed: number;
  skipped: number;
  errors: Array<{ book_id?: string; error: string }>;
}

export interface SynopsisModerationItem {
  moderation_id: string;
  book_id: string;
  book_title: string;
  status: 'pending' | 'accepted' | 'rejected';
  current_synopsis?: string | null;
  proposed_synopsis: string;
  user_synopsis_count: number;
  created_at?: string | null;
  updated_at?: string | null;
  reviewed_at?: string | null;
}

export interface SynopsisModerationListResponse {
  status: string;
  count: number;
  items: SynopsisModerationItem[];
}

class ApiService {
  private async request(endpoint: string, options: RequestInit = {}): Promise<any> {
    const url = `${API_BASE_URL}${endpoint}`;
    const config: RequestInit = {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
    };

    const response = await fetch(url, config);

    if (!response.ok) {
      let errorMessage = `API request failed: ${response.status} ${response.statusText}`;
      try {
        const body = await response.json();
        if (typeof body?.detail === 'string' && body.detail.trim()) {
          errorMessage = body.detail;
        }
      } catch {
        // Keep fallback message when error body is not JSON.
      }

      throw new ApiError(response.status, errorMessage);
    }

    if (response.status === 204) {
      return undefined;
    }

    const text = await response.text();
    return text ? JSON.parse(text) : undefined;
  }

  // Books API
  async getBooks(): Promise<Book[]> {
    return this.request('/books/');
  }

  async getBook(bookId: string): Promise<Book> {
    return this.request(`/books/${bookId}`);
  }

  async getGenres(): Promise<string[]> {
    return this.request('/books/genres');
  }

  // Reviews API
  async getReviewsForBook(bookId: string): Promise<Review[]> {
    return this.request(`/reviews/book/${bookId}`);
  }

  async addReview(accessToken: string, bookId: string, review: ReviewCreate): Promise<Review> {
    return this.request(`/reviews/books/${bookId}`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${accessToken}`,
      },
      body: JSON.stringify(review),
    });
  }

  async updateReview(accessToken: string, reviewId: string, payload: ReviewUpdate): Promise<Review> {
    return this.request(`/reviews/${reviewId}`, {
      method: 'PUT',
      headers: {
        Authorization: `Bearer ${accessToken}`,
      },
      body: JSON.stringify(payload),
    });
  }

  async deleteReview(accessToken: string, reviewId: string): Promise<void> {
    await this.request(`/reviews/${reviewId}`, {
      method: 'DELETE',
      headers: {
        Authorization: `Bearer ${accessToken}`,
      },
    });
  }

  // Chatbot API
  async chat(request: ChatRequest): Promise<ChatResponse> {
    return this.request('/api/chatbot/chat', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  // Auth API
  //Remove slash for /auth/login
  async login(email: string, password: string): Promise<LoginResponse> {
    return this.request('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    });
  }

  async confirmAccount(email: string, confirmationCode: string): Promise<ConfirmAccountResponse> {
    return this.request('/auth/confirm', {
      method: 'POST',
      body: JSON.stringify({
        email,
        confirmation_code: confirmationCode,
      }),
    });
  }

  async register(username: string, email: string, password: string): Promise<{ message: string }> {
    return this.request('/auth/registration', {
      method: 'POST',
      body: JSON.stringify({ username, email, password }),
    });
  }

  // User profile API
  async getMyProfile(accessToken: string): Promise<UserProfile> {
    return this.request('/user-profile/me', {
      headers: {
        Authorization: `Bearer ${accessToken}`,
      },
    });
  }

  async updateMyProfile(accessToken: string, payload: UserProfileUpdate): Promise<UserProfile> {
    return this.request('/user-profile/me', {
      method: 'PATCH',
      headers: {
        Authorization: `Bearer ${accessToken}`,
      },
      body: JSON.stringify(payload),
    });
  }

  // Bookshelf API
  async getMyBookshelf(accessToken: string, status?: ShelfStatus): Promise<BookshelfItem[]> {
    const query = status ? `?status=${status}` : '';
    return this.request(`/bookshelf/${query}`, {
      headers: {
        Authorization: `Bearer ${accessToken}`,
      },
    });
  }

  async addToBookshelf(accessToken: string, bookId: string): Promise<BookshelfItem> {
    return this.request('/bookshelf/', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${accessToken}`,
      },
      body: JSON.stringify({ book_id: bookId }),
    });
  }

  async updateBookshelfStatus(accessToken: string, bookId: string, shelfStatus: ShelfStatus): Promise<BookshelfItem> {
    return this.request(`/bookshelf/${bookId}/status`, {
      method: 'PATCH',
      headers: {
        Authorization: `Bearer ${accessToken}`,
      },
      body: JSON.stringify({ shelf_status: shelfStatus }),
    });
  }

  async removeFromBookshelf(accessToken: string, bookId: string): Promise<void> {
    await this.request(`/bookshelf/${bookId}`, {
      method: 'DELETE',
      headers: {
        Authorization: `Bearer ${accessToken}`,
      },
    });
  }

  async updateBookshelfProgress(accessToken: string, bookId: string, payload: ReadingCheckIn): Promise<BookshelfItem> {
    return this.request(`/bookshelf/${bookId}/progress`, {
      method: 'PATCH',
      headers: {
        Authorization: `Bearer ${accessToken}`,
      },
      body: JSON.stringify(payload),
    });
  }

  async getMyBookshelfStats(accessToken: string): Promise<BookshelfStats> {
    return this.request('/bookshelf/stats', {
      headers: {
        Authorization: `Bearer ${accessToken}`,
      },
    });
  }

  // Admin synopsis moderation API
  async triggerCommunityReviewGeneration(): Promise<SynopsisSyncResult> {
    return this.request('/admin/generate-community-reviews', {
      method: 'POST',
    });
  }

  async getSynopsisModeration(status: 'pending' | 'accepted' | 'rejected' | 'all' = 'pending'): Promise<SynopsisModerationListResponse> {
    return this.request(`/admin/synopsis-moderation?status=${status}`);
  }

  async acceptSynopsisModeration(moderationId: string): Promise<{ status: string; result: { moderation_id: string; book_id: string; status: string; book_title: string; community_synopsis: string } }> {
    return this.request(`/admin/synopsis-moderation/${moderationId}/accept`, {
      method: 'POST',
    });
  }

  async rejectSynopsisModeration(moderationId: string): Promise<{ status: string; result: { moderation_id: string; book_id: string; status: string } }> {
    return this.request(`/admin/synopsis-moderation/${moderationId}/reject`, {
      method: 'POST',
    });
  }
}

export const apiService = new ApiService();