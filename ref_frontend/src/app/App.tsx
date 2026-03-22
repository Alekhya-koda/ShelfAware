import React, { useState } from 'react';
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

  const handleLogin = (admin: boolean, auth?: { accessToken: string; email: string; userId: string }) => {
    setIsAuthenticated(true);
    setIsAdmin(admin);
    setAccessToken(auth?.accessToken ?? null);
    setUserEmail(auth?.email ?? null);
    setUserId(auth?.userId ?? null);
  };

  const handleLogout = () => {
    setIsAuthenticated(false);
    setIsAdmin(false);
    setAccessToken(null);
    setUserEmail(null);
    setUserId(null);
  };

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
          <Route path="/admin" element={isAdmin ? <AdminPanel /> : <Navigate to="/inspiration" replace />} />
          <Route path="*" element={<Navigate to="/inspiration" replace />} />
        </Routes>
      </Layout>
      <Toaster />
    </BrowserRouter>
  );
}
