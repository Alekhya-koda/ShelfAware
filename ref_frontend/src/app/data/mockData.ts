// Mock data for Shelf Aware MVP

import { Book, Review, ReadingProgress, MoodEntry, User } from '../types';

export const mockUser: User = {
  id: '1',
  name: 'Sarah Johnson',
  email: 'sarah@example.com',
  reputation: 4.5,
  role: 'user',
};

export const mockAdminUser: User = {
  id: 'admin1',
  name: 'Admin User',
  email: 'admin@shelfaware.com',
  reputation: 5.0,
  role: 'admin',
};

export const mockBooks: Book[] = [
  {
    id: '1',
    title: 'The Midnight Library',
    author: 'Matt Haig',
    cover: 'https://images.unsplash.com/photo-1543002588-bfa74002ed7e?w=400',
    genre: ['Fantasy', 'Fiction'],
    abstract: 'A dazzling novel about all the choices that go into a life well lived, from the internationally bestselling author of Reasons to Stay Alive and How To Stop Time.',
    averageRating: 4.2,
    totalReviews: 1245,
    price: 15.99,
    condition: 'New',
  },
  {
    id: '2',
    title: 'Atomic Habits',
    author: 'James Clear',
    cover: 'https://images.unsplash.com/photo-1589829085413-56de8ae18c73?w=400',
    genre: ['Psychology/Self-Help', 'Business/Economics', 'Nonfiction'],
    abstract: 'An Easy & Proven Way to Build Good Habits & Break Bad Ones. A supremely practical and useful book.',
    averageRating: 4.8,
    totalReviews: 3421,
    price: 18.99,
    condition: 'New',
  },
  {
    id: '3',
    title: 'Where the Crawdads Sing',
    author: 'Delia Owens',
    cover: 'https://images.unsplash.com/photo-1512820790803-83ca734da794?w=400',
    genre: ['Mystery/Thriller', 'Romance'],
    abstract: 'A story of the forgotten corners of the world, and a girl determined to belong to no one.',
    averageRating: 4.5,
    totalReviews: 2156,
    price: 16.50,
    condition: 'Good',
  },
  {
    id: '4',
    title: 'Educated',
    author: 'Tara Westover',
    cover: 'https://images.unsplash.com/photo-1524995997946-a1c2e315a42f?w=400',
    genre: ['Biography/Memoir', 'Nonfiction'],
    abstract: 'A memoir about a young girl who, kept out of school, leaves her survivalist family and goes on to earn a PhD from Cambridge University.',
    averageRating: 4.7,
    totalReviews: 1876,
    price: 17.99,
    condition: 'New',
  },
  {
    id: '5',
    title: 'The Silent Patient',
    author: 'Alex Michaelides',
    cover: 'https://images.unsplash.com/photo-1544947950-fa07a98d237f?w=400',
    genre: ['Mystery/Thriller', 'Psychology/Self-Help'],
    abstract: 'A woman shoots her husband and then never speaks another word. A psychotherapist becomes obsessed with uncovering her motive.',
    averageRating: 4.1,
    totalReviews: 987,
    price: 14.99,
    condition: 'Good',
  },
  {
    id: '6',
    title: 'Becoming',
    author: 'Michelle Obama',
    cover: 'https://images.unsplash.com/photo-1491841573634-28140fc7ced7?w=400',
    genre: ['Biography/Memoir', 'History', 'Nonfiction'],
    abstract: 'An intimate, powerful, and inspiring memoir by the former First Lady of the United States.',
    averageRating: 4.9,
    totalReviews: 4532,
    price: 19.99,
    condition: 'New',
  },
];

export const mockReadingProgress: ReadingProgress[] = [
  {
    bookId: '1',
    progress: 65,
    startDate: '2026-01-15',
    lastRead: '2026-02-17',
    status: 'reading',
  },
  {
    bookId: '2',
    progress: 100,
    startDate: '2025-12-01',
    lastRead: '2026-01-10',
    status: 'completed',
  },
  {
    bookId: '3',
    progress: 30,
    startDate: '2026-02-01',
    lastRead: '2026-02-15',
    status: 'reading',
  },
  {
    bookId: '5',
    progress: 0,
    startDate: '2026-02-18',
    lastRead: '2026-02-18',
    status: 'want-to-read',
  },
];

export const mockReviews: Review[] = [
  {
    id: 'r1',
    userId: '1',
    bookId: '2',
    rating: 5,
    emotion: ['Inspired', 'Motivated', 'Hopeful'],
    content: 'This book completely changed my perspective on building habits. The 1% improvement philosophy is so powerful and achievable!',
    date: '2026-01-12',
    helpful: 24,
    moderated: true,
  },
  {
    id: 'r2',
    userId: '1',
    bookId: '1',
    rating: 4,
    emotion: ['Thoughtful', 'Emotional', 'Contemplative'],
    content: 'A beautiful exploration of life choices and regret. Made me think deeply about my own decisions.',
    date: '2026-02-10',
    helpful: 12,
    moderated: true,
  },
];

export const mockMoodHistory: MoodEntry[] = [
  {
    date: '2026-02-18',
    mood: 'Curious',
    emotions: ['Adventurous', 'Excited'],
  },
  {
    date: '2026-02-15',
    mood: 'Reflective',
    emotions: ['Thoughtful', 'Calm'],
  },
  {
    date: '2026-02-10',
    mood: 'Anxious',
    emotions: ['Stressed', 'Overwhelmed'],
  },
  {
    date: '2026-02-05',
    mood: 'Happy',
    emotions: ['Joyful', 'Content'],
  },
  {
    date: '2026-02-01',
    mood: 'Motivated',
    emotions: ['Inspired', 'Determined'],
  },
];

export const emotionTags = [
  'Happy',
  'Sad',
  'Angry',
  'Excited',
  'Scared',
  'Romantic',
  'Suspenseful',
  'Dark',
  'Hopeful',
  'Nostalgic',
  'Peaceful',
  'Curious',
  'Empowered',
  'Lonely',
  'Grateful',
  'Confused',
  'Inspired',
  'Amused',
  'Moved',
  'Adventurous',
  'Reflective',
  'Whimsical',
  'Heartbroken',
  'Triumphant',
];