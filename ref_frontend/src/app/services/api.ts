// API service for connecting to the ShelfAware backend
const API_BASE_URL = 'http://localhost:8000';

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
  rating: number;
  comment?: string;
  mood?: string;
  created_at: string;
}

export interface ReviewCreate {
  rating: number;
  comment?: string;
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

  // Chatbot API
  async chat(request: ChatRequest): Promise<ChatResponse> {
    return this.request('/api/chatbot/chat', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  // Auth API
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
}

export const apiService = new ApiService();