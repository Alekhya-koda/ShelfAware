import React, { useState } from 'react';
import { useNavigate } from 'react-router';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { toast } from 'sonner';
import { apiService, ApiError } from '../services/api';
import logoImage from '../../assets/08044afe8eb8f9700793bdbb3ce5779e85ca56f7.png';

interface LoginProps {
  onLogin: (isAdmin: boolean, auth?: { accessToken: string; email: string; userId: string }) => void;
}

export function Login({ onLogin }: LoginProps) {
  const navigate = useNavigate();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isCreatingAccount, setIsCreatingAccount] = useState(false);
  const [isSigningIn, setIsSigningIn] = useState(false);
  const [showConfirmAccount, setShowConfirmAccount] = useState(false);
  const [confirmationCode, setConfirmationCode] = useState('');
  const [isConfirmingAccount, setIsConfirmingAccount] = useState(false);

  const handleLogin = async (e: React.FormEvent, isAdmin: boolean = false) => {
    e.preventDefault();
    if (isAdmin) {
      onLogin(true);
      navigate('/admin');
      return;
    }

    const normalizedEmail = email.trim().toLowerCase();
    if (!normalizedEmail || !password.trim()) {
      toast.error('Please enter email and password');
      return;
    }

    try {
      setIsSigningIn(true);
      const result = await apiService.login(normalizedEmail, password);
      const accessToken = result.tokens?.access_token;
      if (!accessToken) {
        throw new Error('Login response did not include an access token');
      }

      onLogin(false, { accessToken, email: result.user.email, userId: result.user.user_id });
      navigate('/inspiration');
    } catch (error) {
      console.error('Login failed:', error);
      if (error instanceof ApiError && error.status === 403 && error.message.toLowerCase().includes('not confirmed')) {
        toast.error('Account not confirmed. Check your email for the confirmation code, then confirm before signing in.');
        setShowConfirmAccount(true);
      } else {
        toast.error(error instanceof Error ? error.message : 'Failed to sign in');
      }
    } finally {
      setIsSigningIn(false);
    }
  };

  const handleConfirmAccount = async () => {
    const normalizedEmail = email.trim().toLowerCase();
    const code = confirmationCode.trim();

    if (!normalizedEmail || !code) {
      toast.error('Enter email and confirmation code');
      return;
    }

    try {
      setIsConfirmingAccount(true);
      const result = await apiService.confirmAccount(normalizedEmail, code);
      toast.success(result.message || 'Account confirmed. You can now sign in.');
      setShowConfirmAccount(false);
      setConfirmationCode('');
    } catch (error) {
      console.error('Account confirmation failed:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to confirm account');
    } finally {
      setIsConfirmingAccount(false);
    }
  };

  const handleCreateAccount = async () => {
    const normalizedEmail = email.trim().toLowerCase();
    if (!normalizedEmail || !password.trim()) {
      toast.error('Enter email and password to create an account');
      return;
    }

    const usernameFromEmail = normalizedEmail.split('@')[0]?.replace(/[^a-zA-Z0-9_.-]/g, '') || '';
    const username = usernameFromEmail.length >= 3
      ? usernameFromEmail.slice(0, 50)
      : `user${Date.now().toString().slice(-8)}`;

    try {
      setIsCreatingAccount(true);
      const response = await fetch('http://localhost:8000/auth/registration', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          username,
          email: normalizedEmail,
          password,
        }),
      });

      const body = await response.json().catch(() => ({}));
      if (!response.ok) {
        const detail = typeof body?.detail === 'string' ? body.detail : 'Failed to create account';
        throw new Error(detail);
      }

      toast.success('Account created. Check email for confirmation if required.');
    } catch (error) {
      console.error('Create account failed:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to create account');
    } finally {
      setIsCreatingAccount(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-purple-50 to-blue-50 p-4">
      <div className="w-full max-w-md">
        <div className="flex items-center justify-center mb-8">
          <img src={logoImage} alt="Shelf Aware" className="w-64" />
        </div>

        <Tabs defaultValue="user" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="user">User Login</TabsTrigger>
            <TabsTrigger value="admin">Admin Login</TabsTrigger>
          </TabsList>

          <TabsContent value="user">
            <Card>
              <CardHeader>
                <CardTitle>Welcome Back</CardTitle>
                <CardDescription>
                  Sign in to discover books that match your emotions
                </CardDescription>
              </CardHeader>
              <CardContent>
                <form onSubmit={(e) => handleLogin(e, false)} className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="user-email">Email</Label>
                    <Input
                      id="user-email"
                      type="email"
                      placeholder="sarah@example.com"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="user-password">Password</Label>
                    <Input
                      id="user-password"
                      type="password"
                      placeholder="••••••••"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                    />
                  </div>
                  <Button
                    type="submit"
                    className="w-full bg-purple-600 hover:bg-purple-700 text-white"
                    disabled={isSigningIn}
                  >
                    {isSigningIn ? 'Signing In...' : 'Sign In'}
                  </Button>
                  <Button
                    type="button"
                    className="w-full bg-black hover:bg-black/90 text-white"
                    variant="default"
                    onClick={handleCreateAccount}
                    disabled={isCreatingAccount}
                  >
                    {isCreatingAccount ? 'Creating Account...' : 'Create Account'}
                  </Button>
                  {showConfirmAccount && (
                    <div className="mt-4 border rounded-lg p-3 bg-amber-50 space-y-3">
                      <p className="text-sm font-medium text-amber-900">
                        Account confirmation required
                      </p>
                      <Input
                        value={confirmationCode}
                        onChange={(e) => setConfirmationCode(e.target.value)}
                        placeholder="Enter confirmation code"
                      />
                      <Button
                        type="button"
                        className="w-full"
                        variant="outline"
                        onClick={handleConfirmAccount}
                        disabled={isConfirmingAccount}
                      >
                        {isConfirmingAccount ? 'Confirming...' : 'Confirm Account'}
                      </Button>
                    </div>
                  )}
                </form>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="admin">
            <Card>
              <CardHeader>
                <CardTitle>Admin Access</CardTitle>
                <CardDescription>
                  Manage book data and moderate reviews
                </CardDescription>
              </CardHeader>
              <CardContent>
                <form onSubmit={(e) => handleLogin(e, true)} className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="admin-email">Admin Email</Label>
                    <Input
                      id="admin-email"
                      type="email"
                      placeholder="admin@shelfaware.com"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="admin-password">Password</Label>
                    <Input
                      id="admin-password"
                      type="password"
                      placeholder="••••••••"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                    />
                  </div>
                  <Button type="submit" className="w-full" variant="default">
                    Admin Sign In
                  </Button>
                  <p className="text-sm text-center text-gray-600">
                    Demo: Click to login as admin
                  </p>
                </form>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}