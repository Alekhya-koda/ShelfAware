import React, { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router';
import { Toaster } from './components/ui/sonner';
import { Login } from './components/Login';
import { Layout } from './components/Layout';
import { Inspiration } from './components/Inspiration';
import { Bookshelf } from './components/Bookshelf';
import { BookDetail } from './components/BookDetail';
import { Profile } from './components/Profile';
import { Chatbot } from './components/Chatbot';
import { AdminPanel } from './components/AdminPanel';

export default function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isAdmin, setIsAdmin] = useState(false);
  const [accessToken, setAccessToken] = useState<string | null>(null);
  const [userEmail, setUserEmail] = useState<string | null>(null);
  const [userId, setUserId] = useState<string | null>(null);
  const [isLoadingAuth, setIsLoadingAuth] = useState(true);

  // Load auth state from localStorage on component mount
  useEffect(() => {
    const savedAuth = localStorage.getItem('auth_state');
    if (savedAuth) {
      try {
        const authState = JSON.parse(savedAuth);
        setIsAuthenticated(authState.isAuthenticated);
        setIsAdmin(authState.isAdmin);
        setAccessToken(authState.accessToken);
        setUserEmail(authState.userEmail);
        setUserId(authState.userId);
      } catch (error) {
        console.error('Failed to load auth state:', error);
        localStorage.removeItem('auth_state');
      }
    }
    setIsLoadingAuth(false);
  }, []);

  const handleLogin = (admin: boolean, auth?: { accessToken: string; email: string; userId: string }) => {
    setIsAuthenticated(true);
    setIsAdmin(admin);
    setAccessToken(auth?.accessToken ?? null);
    setUserEmail(auth?.email ?? null);
    setUserId(auth?.userId ?? null);

    // Persist auth state to localStorage
    localStorage.setItem('auth_state', JSON.stringify({
      isAuthenticated: true,
      isAdmin: admin,
      accessToken: auth?.accessToken ?? null,
      userEmail: auth?.email ?? null,
      userId: auth?.userId ?? null,
    }));
  };

  const handleLogout = () => {
    setIsAuthenticated(false);
    setIsAdmin(false);
    setAccessToken(null);
    setUserEmail(null);
    setUserId(null);
    localStorage.removeItem('auth_state');
  };

  if (isLoadingAuth) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-purple-50 to-blue-50">
        <div className="text-center">
          <p className="text-gray-600">Loading...</p>
        </div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return (
      <BrowserRouter>
        <Login onLogin={handleLogin} />
        <Toaster />
      </BrowserRouter>
    );
  }

  return (
    <BrowserRouter>
      <Layout isAdmin={isAdmin} onLogout={handleLogout}>
        <Routes>
          <Route path="/" element={<Navigate to="/inspiration" replace />} />
          <Route path="/inspiration" element={<Inspiration accessToken={accessToken} />} />
          <Route path="/bookshelf" element={<Bookshelf accessToken={accessToken} />} />
          <Route path="/book/:bookId" element={<BookDetail accessToken={accessToken} userId={userId} />} />
          <Route path="/profile" element={<Profile accessToken={accessToken} userEmail={userEmail} userId={userId} />} />
          <Route path="/chatbot" element={<Chatbot userId={userId} />} />
          <Route path="/admin" element={isAdmin ? <AdminPanel accessToken={accessToken} /> : <Navigate to="/inspiration" replace />} />
          <Route path="*" element={<Navigate to="/inspiration" replace />} />
        </Routes>
      </Layout>
      <Toaster />
    </BrowserRouter>
  );
}
