import React, { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Avatar, AvatarFallback } from './ui/avatar';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Input } from './ui/input';
import { Textarea } from './ui/textarea';
import { Button } from './ui/button';
import { Checkbox } from './ui/checkbox';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Star, BookOpen, Award, TrendingUp, Calendar, Heart } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line } from 'recharts';
import { mockBooks, mockMoodHistory } from '../data/mockData';
import { apiService, Book, BookshelfItem, Review } from '../services/api';
import { toast } from 'sonner';

interface ProfileProps {
  accessToken: string | null;
  userEmail: string | null;
  userId: string | null;
}

const COUNTRIES = [
  'Australia',
  'Brazil',
  'Canada',
  'China',
  'France',
  'Germany',
  'India',
  'Indonesia',
  'Japan',
  'Malaysia',
  'Mexico',
  'Netherlands',
  'New Zealand',
  'Philippines',
  'Singapore',
  'South Korea',
  'Spain',
  'Thailand',
  'United Arab Emirates',
  'United Kingdom',
  'United States',
  'Vietnam',
];

function parseFavoriteGenres(raw: string): string[] {
  const trimmed = raw.trim();
  if (!trimmed) return [];

  try {
    const parsed = JSON.parse(trimmed);
    if (Array.isArray(parsed)) {
      return parsed.map((genre) => String(genre).trim()).filter(Boolean);
    }
  } catch {
    // Support legacy comma-separated values.
  }

  return trimmed
    .split(',')
    .map((genre) => genre.trim())
    .filter(Boolean);
}

export function Profile({ accessToken, userEmail, userId }: ProfileProps) {
  const navigate = useNavigate();
  const [isProfileLoading, setIsProfileLoading] = useState(true);
  const [isSavingProfile, setIsSavingProfile] = useState(false);
  const [isActivityLoading, setIsActivityLoading] = useState(true);
  const [isEditingProfile, setIsEditingProfile] = useState(false);
  const [isLoadingGenres, setIsLoadingGenres] = useState(false);
  const [booksById, setBooksById] = useState<Record<string, Book>>({});
  const [myShelfItems, setMyShelfItems] = useState<BookshelfItem[]>([]);
  const [myReviews, setMyReviews] = useState<Review[]>([]);
  const [availableGenres, setAvailableGenres] = useState<string[]>([]);
  const [profileForm, setProfileForm] = useState({
    display_name: '',
    profile_photo_url: '',
    bio: '',
    location: '',
    favorite_genres_json: '',
  });
  const [profileViewSnapshot, setProfileViewSnapshot] = useState({
    display_name: '',
    profile_photo_url: '',
    bio: '',
    location: '',
    favorite_genres_json: '',
  });

  useEffect(() => {
    const loadMyProfile = async () => {
      if (!accessToken) {
        setIsProfileLoading(false);
        return;
      }

      try {
        const profile = await apiService.getMyProfile(accessToken);
        setProfileForm({
          display_name: profile.display_name || '',
          profile_photo_url: profile.profile_photo_url || '',
          bio: profile.bio || '',
          location: profile.location || '',
          favorite_genres_json: profile.favorite_genres_json || '',
        });
        setProfileViewSnapshot({
          display_name: profile.display_name || '',
          profile_photo_url: profile.profile_photo_url || '',
          bio: profile.bio || '',
          location: profile.location || '',
          favorite_genres_json: profile.favorite_genres_json || '',
        });
      } catch (error) {
        console.error('Failed to load profile:', error);
        toast.error('Could not load user profile');
      } finally {
        setIsProfileLoading(false);
      }
    };

    loadMyProfile();
  }, [accessToken]);

  useEffect(() => {
    const loadActivity = async () => {
      if (!accessToken) {
        setIsActivityLoading(false);
        return;
      }

      try {
        setIsActivityLoading(true);

        const [allBooks, shelfItems] = await Promise.all([
          apiService.getBooks(),
          apiService.getMyBookshelf(accessToken),
        ]);

        const byId = allBooks.reduce<Record<string, Book>>((acc, book) => {
          acc[book.book_id] = book;
          return acc;
        }, {});

        setBooksById(byId);
        setMyShelfItems(shelfItems);

        if (userId && shelfItems.length > 0) {
          const uniqueBookIds = Array.from(new Set(shelfItems.map((item) => item.book_id)));
          const reviewsByBook = await Promise.all(
            uniqueBookIds.map((bookId) => apiService.getReviewsForBook(bookId).catch(() => [] as Review[]))
          );

          const mine = reviewsByBook
            .flat()
            .filter((review) => String(review.user_id || '') === String(userId))
            .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());

          setMyReviews(mine);
        } else {
          setMyReviews([]);
        }
      } catch (error) {
        console.error('Failed to load profile activity:', error);
        toast.error('Could not load reading activity from backend');
      } finally {
        setIsActivityLoading(false);
      }
    };

    loadActivity();
  }, [accessToken, userId]);

  useEffect(() => {
    const loadGenres = async () => {
      try {
        setIsLoadingGenres(true);
        const genres = await apiService.getGenres();
        setAvailableGenres(genres);
      } catch (error) {
        console.error('Failed to load genres:', error);
        setAvailableGenres([]);
      } finally {
        setIsLoadingGenres(false);
      }
    };

    loadGenres();
  }, []);

  const userInitials = useMemo(() => {
    const source = profileForm.display_name || userEmail || 'User';
    return source
      .split(' ')
      .filter(Boolean)
      .slice(0, 2)
      .map((part) => part[0]?.toUpperCase() || '')
      .join('') || 'U';
  }, [profileForm.display_name, userEmail]);

  const handleProfileFieldChange = (field: keyof typeof profileForm, value: string) => {
    setProfileForm((prev) => ({ ...prev, [field]: value }));
  };

  const selectedFavoriteGenres = useMemo(
    () => parseFavoriteGenres(profileForm.favorite_genres_json),
    [profileForm.favorite_genres_json]
  );

  const handleGenreToggle = (genre: string, checked: boolean) => {
    const nextGenres = checked
      ? Array.from(new Set([...selectedFavoriteGenres, genre]))
      : selectedFavoriteGenres.filter((item) => item !== genre);
    setProfileForm((prev) => ({
      ...prev,
      favorite_genres_json: JSON.stringify(nextGenres),
    }));
  };

  const handleStartEditingProfile = () => {
    setProfileForm(profileViewSnapshot);
    setIsEditingProfile(true);
  };

  const handleCancelEditingProfile = () => {
    setProfileForm(profileViewSnapshot);
    setIsEditingProfile(false);
  };

  const handleSaveProfile = async () => {
    if (!accessToken) {
      toast.error('Please sign in to update your profile');
      return;
    }

    if (!profileForm.display_name.trim()) {
      toast.error('Display name is required');
      return;
    }

    try {
      setIsSavingProfile(true);
      const updated = await apiService.updateMyProfile(accessToken, {
        display_name: profileForm.display_name.trim(),
        profile_photo_url: profileForm.profile_photo_url?.trim() || null,
        bio: profileForm.bio?.trim() || null,
        location: profileForm.location?.trim() || null,
        favorite_genres_json: profileForm.favorite_genres_json?.trim() || null,
      });

      setProfileForm({
        display_name: updated.display_name || '',
        profile_photo_url: updated.profile_photo_url || '',
        bio: updated.bio || '',
        location: updated.location || '',
        favorite_genres_json: updated.favorite_genres_json || '',
      });
      setProfileViewSnapshot({
        display_name: updated.display_name || '',
        profile_photo_url: updated.profile_photo_url || '',
        bio: updated.bio || '',
        location: updated.location || '',
        favorite_genres_json: updated.favorite_genres_json || '',
      });
      setIsEditingProfile(false);

      toast.success('Profile updated');
    } catch (error) {
      console.error('Failed to update profile:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to update profile');
    } finally {
      setIsSavingProfile(false);
    }
  };

  const totalBooks = myShelfItems.length;
  const completedBooks = myShelfItems.filter((p) => p.shelf_status === 'read').length;
  const currentlyReading = myShelfItems.filter((p) => p.shelf_status === 'currently_reading').length;

  // No backend endpoint currently exposes profile genre analytics, so keep this mock chart.
  const genreData = mockBooks.reduce((acc, book: any) => {
    const genres = Array.isArray(book?.genre) ? book.genre : [];
    genres.forEach((genre: string) => {
      acc[genre] = (acc[genre] || 0) + 1;
    });
    return acc;
  }, {} as Record<string, number>);

  const genreChartData = Object.entries(genreData).map(([genre, count]) => ({
    genre,
    count,
  }));

  // Mood history chart data
  const moodChartData = mockMoodHistory.map((entry) => ({
    date: new Date(entry.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
    mood: entry.mood,
  }));

  // Average rating given
  const avgRatingGiven =
    myReviews.reduce((sum, r) => sum + r.rating, 0) / myReviews.length || 0;

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-gray-900">My Profile</h1>

      {/* User Info Card */}
      <Card>
        <CardContent className="p-6">
          <div className="flex flex-col gap-6 lg:flex-row lg:items-start lg:space-x-6">
            <Avatar className="size-24">
              {profileForm.profile_photo_url ? (
                <img
                  src={profileForm.profile_photo_url}
                  alt={profileForm.display_name || 'Profile photo'}
                  className="w-full h-full object-cover"
                />
              ) : (
                <AvatarFallback className="text-2xl">{userInitials}</AvatarFallback>
              )}
            </Avatar>
            <div className="flex-1">
              <h2 className="text-2xl font-bold mb-1">
                {isProfileLoading ? 'Loading profile...' : profileForm.display_name || 'Set your display name'}
              </h2>
              <p className="text-gray-600 mb-3">{userEmail || '-'}</p>
              <div className="flex items-center space-x-6">
                <div className="flex items-center">
                  <Star className="size-5 text-yellow-500 mr-2 fill-current" />
                  <span className="font-semibold">-</span>
                  <span className="text-sm text-gray-600 ml-1">Reputation</span>
                </div>
                <div className="flex items-center">
                  <Award className="size-5 text-purple-500 mr-2" />
                  <span className="font-semibold">{myReviews.length}</span>
                  <span className="text-sm text-gray-600 ml-1">Reviews</span>
                </div>
                <div className="flex items-center">
                  <BookOpen className="size-5 text-blue-500 mr-2" />
                  <span className="font-semibold">{totalBooks}</span>
                  <span className="text-sm text-gray-600 ml-1">Books</span>
                </div>
              </div>

              {isEditingProfile ? (
                <>
                  <div className="mt-5 grid md:grid-cols-2 gap-3">
                    <div className="space-y-1">
                      <label className="text-sm font-medium text-gray-700">Display Name</label>
                      <Input
                        value={profileForm.display_name}
                        placeholder="Enter display name"
                        onChange={(e) => handleProfileFieldChange('display_name', e.target.value)}
                      />
                    </div>
                    <div className="space-y-1">
                      <label className="text-sm font-medium text-gray-700">Location</label>
                      <Select
                        value={profileForm.location || '__none'}
                        onValueChange={(value) => handleProfileFieldChange('location', value === '__none' ? '' : value)}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Select a country" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="__none">Not specified</SelectItem>
                          {COUNTRIES.map((country) => (
                            <SelectItem key={country} value={country}>
                              {country}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-1 md:col-span-2">
                      <label className="text-sm font-medium text-gray-700">Profile Photo URL</label>
                      <Input
                        value={profileForm.profile_photo_url}
                        placeholder="https://..."
                        onChange={(e) => handleProfileFieldChange('profile_photo_url', e.target.value)}
                      />
                    </div>
                    <div className="space-y-1 md:col-span-2">
                      <label className="text-sm font-medium text-gray-700">Bio</label>
                      <Textarea
                        value={profileForm.bio}
                        placeholder="Tell others what you like to read"
                        onChange={(e) => handleProfileFieldChange('bio', e.target.value)}
                      />
                    </div>
                    <div className="space-y-1 md:col-span-2">
                      <label className="text-sm font-medium text-gray-700">Favorite Genres</label>
                      {isLoadingGenres ? (
                        <p className="text-sm text-gray-500">Loading genres...</p>
                      ) : availableGenres.length === 0 ? (
                        <p className="text-sm text-gray-500">No genres available from database.</p>
                      ) : (
                        <div className="space-y-3">
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-2 max-h-44 overflow-y-auto rounded-md border p-3">
                            {availableGenres.map((genre) => {
                              const isChecked = selectedFavoriteGenres.includes(genre);
                              return (
                                <label key={genre} className="flex items-center gap-2 text-sm text-gray-700 cursor-pointer">
                                  <Checkbox
                                    checked={isChecked}
                                    onCheckedChange={(checked) => handleGenreToggle(genre, Boolean(checked))}
                                  />
                                  <span>{genre}</span>
                                </label>
                              );
                            })}
                          </div>
                          <div className="flex flex-wrap gap-2">
                            {selectedFavoriteGenres.length > 0 ? (
                              selectedFavoriteGenres.map((genre) => (
                                <Badge key={genre} variant="secondary">{genre}</Badge>
                              ))
                            ) : (
                              <span className="text-sm text-gray-500">No genres selected.</span>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="mt-4 flex gap-2">
                    <Button onClick={handleSaveProfile} disabled={isSavingProfile || isProfileLoading}>
                      {isSavingProfile ? 'Saving...' : 'Save Profile Details'}
                    </Button>
                    <Button variant="secondary" onClick={handleCancelEditingProfile} disabled={isSavingProfile}>
                      Cancel
                    </Button>
                  </div>
                </>
              ) : (
                <>
                  <div className="mt-5 grid md:grid-cols-2 gap-3 text-sm">
                    <div>
                      <p className="text-gray-500">Location</p>
                      <p className="font-medium text-gray-900">{profileForm.location || '-'}</p>
                    </div>
                    <div className="md:col-span-2">
                      <p className="text-gray-500">Bio</p>
                      <p className="font-medium text-gray-900 whitespace-pre-wrap">{profileForm.bio || '-'}</p>
                    </div>
                    <div className="md:col-span-2">
                      <p className="text-gray-500">Favorite Genres</p>
                      <div className="mt-1 flex flex-wrap gap-2">
                        {selectedFavoriteGenres.length > 0 ? (
                          selectedFavoriteGenres.map((genre) => (
                            <Badge key={genre} variant="secondary">{genre}</Badge>
                          ))
                        ) : (
                          <p className="font-medium text-gray-900">-</p>
                        )}
                      </div>
                    </div>
                  </div>

                  <div className="mt-4">
                    <Button onClick={handleStartEditingProfile} disabled={isProfileLoading}>
                      Update Profile
                    </Button>
                  </div>
                </>
              )}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Reading Statistics */}
      <div className="grid md:grid-cols-3 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <BookOpen className="size-5 mr-2 text-blue-500" />
              Currently Reading
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-blue-600">{isActivityLoading ? '-' : currentlyReading}</div>
            <p className="text-sm text-gray-600 mt-1">books in progress</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Star className="size-5 mr-2 text-yellow-500" />
              Completed
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-yellow-600">{isActivityLoading ? '-' : completedBooks}</div>
            <p className="text-sm text-gray-600 mt-1">books finished</p>
            <Progress value={totalBooks > 0 ? (completedBooks / totalBooks) * 100 : 0} className="mt-3" />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <TrendingUp className="size-5 mr-2 text-green-500" />
              Avg Rating Given
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-green-600">{isActivityLoading ? '-' : avgRatingGiven.toFixed(1)}</div>
            <div className="flex mt-2">
              {Array.from({ length: 5 }).map((_, i) => (
                <Star
                  key={i}
                  className={`size-4 ${
                    i < Math.round(avgRatingGiven)
                      ? 'text-yellow-500 fill-current'
                      : 'text-gray-300'
                  }`}
                />
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Charts and Analytics */}
      <Tabs defaultValue="genres" className="w-full">
        <TabsList>
          <TabsTrigger value="genres">Reading by Genre</TabsTrigger>
          <TabsTrigger value="mood">Mood History</TabsTrigger>
          <TabsTrigger value="reviews">My Reviews</TabsTrigger>
        </TabsList>

        <TabsContent value="genres" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>Books by Genre</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={genreChartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="genre" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="count" fill="#8b5cf6" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="mood" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>Mood Tracking History</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {mockMoodHistory.map((entry, index) => (
                  <div key={index} className="flex items-center justify-between border-b pb-3 last:border-b-0">
                    <div className="flex items-center">
                      <Calendar className="size-4 text-gray-400 mr-3" />
                      <div>
                        <div className="font-semibold">{entry.mood}</div>
                        <div className="text-sm text-gray-600">
                          {new Date(entry.date).toLocaleDateString()}
                        </div>
                      </div>
                    </div>
                    <div className="flex flex-wrap gap-2 justify-end">
                      {entry.emotions.map((emotion) => (
                        <Badge key={emotion} variant="secondary" className="text-xs">
                          {emotion}
                        </Badge>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="reviews" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>My Reviews ({myReviews.length})</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {isActivityLoading ? (
                <p className="text-center text-gray-500 py-6">Loading your reviews...</p>
              ) : myReviews.length === 0 ? (
                <p className="text-center text-gray-500 py-6">No reviews found yet.</p>
              ) : myReviews.map((review) => {
                const book = booksById[review.book_id];
                const reviewMoodText = (review.book_mood || review.mood || '')
                  .split(',')
                  .map((m) => m.trim())
                  .filter(Boolean);
                return (
                  <div
                    key={review.review_id}
                    className="border rounded-lg p-4 cursor-pointer hover:bg-gray-50 transition-colors"
                    onClick={() => navigate(`/book/${review.book_id}`)}
                    role="button"
                    tabIndex={0}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        navigate(`/book/${review.book_id}`);
                      }
                    }}
                  >
                    <div className="flex items-start justify-between mb-2">
                      <div>
                        <h4 className="font-semibold">{book?.title || review.title || review.book_id}</h4>
                        <div className="flex items-center mt-1">
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
                        </div>
                      </div>
                      <span className="text-xs text-gray-500">
                        {new Date(review.created_at).toLocaleDateString()}
                      </span>
                    </div>
                    <div className="flex flex-wrap gap-2 mb-2">
                      {reviewMoodText.map((emotion) => (
                        <Badge key={emotion} variant="outline" className="text-xs">
                          <Heart className="size-3 mr-1" />
                          {emotion}
                        </Badge>
                      ))}
                    </div>
                    <p className="text-gray-700">{review.comment || '-'}</p>
                  </div>
                );
              })}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
