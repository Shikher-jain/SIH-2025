import { createContext, useContext, useState, useEffect, type ReactNode } from 'react';
import { AuthAPI, type User } from '../services/authApi';
import { GuestModeService } from '../services/guestModeService';
import { GuestFarmStorage } from '../utils/guestFarmStorage';

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  isGuestMode: boolean;
  login: (email: string, password: string) => Promise<{ success: boolean; error?: string }>;
  register: (email: string, password: string, name: string, phone: string, address: string) => Promise<{ success: boolean; error?: string }>;
  logout: () => void;
  migrationStatus: {
    isLoading: boolean;
    result: { success: boolean; migratedCount: number; errors: string[] } | null;
  };
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isGuestMode, setIsGuestMode] = useState(false);
  const [migrationStatus, setMigrationStatus] = useState<{
    isLoading: boolean;
    result: { success: boolean; migratedCount: number; errors: string[] } | null;
  }>({
    isLoading: false,
    result: null,
  });

  useEffect(() => {
    // Check for stored user session and verify token
    const initAuth = async () => {
      console.log('🔍 Initializing authentication...');
      
      try {
        const isAuth = AuthAPI.isAuthenticated();
        console.log('🔑 Is authenticated:', isAuth);
        
        if (isAuth) {
          const currentUser = AuthAPI.getCurrentUser();
          console.log('👤 Current user from storage:', currentUser);
          
          if (currentUser) {
            setUser(currentUser);
            setIsGuestMode(false);
            console.log('✅ User set in context:', currentUser);
            
            // Optionally verify token with backend
            try {
              await AuthAPI.verifyToken();
              console.log('✅ Token verified successfully');
            } catch (error) {
              console.error('❌ Token verification failed:', error);
              // Clear invalid session
              AuthAPI.logout();
              setUser(null);
              setIsGuestMode(false);
            }
          } else {
            console.log('❌ No current user found in storage');
          }
        } else {
          // Check for guest mode
          const isGuest = GuestModeService.isGuestMode();
          console.log('🔍 Checking guest mode:', isGuest);
          
          if (isGuest) {
            const guestUser = GuestModeService.getGuestUser();
            if (guestUser) {
              setUser(guestUser);
              setIsGuestMode(true);
              console.log('👻 Guest user set in context:', guestUser);
            }
          } else {
            // Auto-enable guest mode for new visitors
            if (GuestModeService.shouldAutoEnableGuest()) {
              console.log('🎯 Auto-enabling guest mode for new visitor');
              const guestUser = GuestModeService.enableGuestMode();
              setUser(guestUser);
              setIsGuestMode(true);
              console.log('👻 Auto guest user created:', guestUser);
            }
          }
        }
      } catch (error) {
        console.error('❌ Error during auth initialization:', error);
      } finally {
        setIsLoading(false);
        console.log('🏁 Auth initialization complete');
      }
    };

    initAuth();
  }, []);

  const login = async (email: string, password: string) => {
    try {
      console.log('🔐 Attempting login for:', email);
      
      // Check if we need to migrate guest farms
      const hasGuestFarms = GuestModeService.hasGuestFarmsToMigrate();
      
      const response = await AuthAPI.login({ email, password });
      console.log('📥 Login response:', response);
      
      if (response.code === 1 && response.result?.user) {
        setUser(response.result.user);
        setIsGuestMode(false);
        console.log('✅ Login successful, user set:', response.result.user);
        
        // Migrate guest farms if any exist
        if (hasGuestFarms) {
          console.log('🔄 Starting guest farm migration after login...');
          setMigrationStatus({ isLoading: true, result: null });
          
          try {
            const migrationResult = await GuestModeService.migrateGuestFarmsToUser();
            setMigrationStatus({ isLoading: false, result: migrationResult });
            
            if (migrationResult.success) {
              console.log(`✅ Successfully migrated ${migrationResult.migratedCount} guest farms`);
            } else {
              console.error('❌ Farm migration completed with errors:', migrationResult.errors);
            }
          } catch (migrationError) {
            console.error('❌ Farm migration failed:', migrationError);
            setMigrationStatus({ 
              isLoading: false, 
              result: { success: false, migratedCount: 0, errors: ['Migration failed'] } 
            });
          }
        }
        
        return { success: true };
      } else {
        console.log('❌ Login failed:', response.message);
        return { success: false, error: response.message || 'Invalid email or password' };
      }
    } catch (error: any) {
      console.error('❌ Login error:', error);
      return { 
        success: false, 
        error: error.message || 'Login failed. Please try again.' 
      };
    }
  };

  const register = async (email: string, password: string, name: string, phone: string, address: string) => {
    try {
      console.log('📝 Attempting registration for:', email);
      
      // Check if we need to migrate guest farms
      const hasGuestFarms = GuestModeService.hasGuestFarmsToMigrate();
      
      const response = await AuthAPI.register({
        fullName: name,
        email,
        password,
        phone,
        address
      });
      console.log('📥 Registration response:', response);

      if (response.code === 1 && response.result?.user) {
        setUser(response.result.user);
        setIsGuestMode(false);
        console.log('✅ Registration successful, user set:', response.result.user);
        
        // Migrate guest farms if any exist
        if (hasGuestFarms) {
          console.log('🔄 Starting guest farm migration after registration...');
          setMigrationStatus({ isLoading: true, result: null });
          
          try {
            const migrationResult = await GuestModeService.migrateGuestFarmsToUser();
            setMigrationStatus({ isLoading: false, result: migrationResult });
            
            if (migrationResult.success) {
              console.log(`✅ Successfully migrated ${migrationResult.migratedCount} guest farms`);
            } else {
              console.error('❌ Farm migration completed with errors:', migrationResult.errors);
            }
          } catch (migrationError) {
            console.error('❌ Farm migration failed:', migrationError);
            setMigrationStatus({ 
              isLoading: false, 
              result: { success: false, migratedCount: 0, errors: ['Migration failed'] } 
            });
          }
        }
        
        return { success: true };
      } else {
        console.log('❌ Registration failed:', response.message);
        return { success: false, error: response.message || 'Registration failed' };
      }
    } catch (error: any) {
      console.error('❌ Registration error:', error);
      return { 
        success: false, 
        error: error.message || 'Registration failed. Please try again.' 
      };
    }
  };

  const logout = async () => {
    try {
      console.log('🚪 Logging out...');
      
      if (isGuestMode) {
        // For guest mode, clear guest farm data and guest session
        GuestFarmStorage.clearAllFarms();
        GuestModeService.disableGuestMode();
        const newGuestUser = GuestModeService.enableGuestMode();
        setUser(newGuestUser);
        setIsGuestMode(true);
        console.log('👻 Guest logout: created new guest session and cleared guest farms');
        // Reset migration status
        setMigrationStatus({ isLoading: false, result: null });
        // Redirect to dashboard to show the clean guest interface
        window.location.href = '/dashboard';
      } else {
        // For authenticated users, perform full logout
        await AuthAPI.logout();
        console.log('✅ Logout successful');
        
        setUser(null);
        setIsGuestMode(false);
        setMigrationStatus({ isLoading: false, result: null });
        console.log('👤 User cleared from context');
        
        // Auto-enable guest mode for the logged out user
        const guestUser = GuestModeService.enableGuestMode();
        setUser(guestUser);
        setIsGuestMode(true);
        console.log('👻 Auto-enabled guest mode after logout');
        
        // Redirect to dashboard to show guest mode
        window.location.href = '/dashboard';
      }
    } catch (error) {
      console.error('❌ Logout error:', error);
      // Fallback: clear everything and enable guest mode
      setUser(null);
      setIsGuestMode(false);
      setMigrationStatus({ isLoading: false, result: null });
      
      const guestUser = GuestModeService.enableGuestMode();
      setUser(guestUser);
      setIsGuestMode(true);
      
      window.location.href = '/dashboard';
    }
  };

  // Debug current auth state
  console.log('🔍 Current auth state:', {
    user: user,
    isAuthenticated: !!user,
    isLoading: isLoading,
    isGuestMode: isGuestMode,
    migrationStatus: migrationStatus
  });

  const value = {
    user,
    isAuthenticated: !!user,
    isLoading,
    isGuestMode,
    login,
    register,
    logout,
    migrationStatus
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}

// Enhanced Protected Route Component with debugging
interface ProtectedRouteProps {
  children: ReactNode;
  requiredRole?: 'admin' | 'user';
}

export function ProtectedRoute({ children, requiredRole }: ProtectedRouteProps) {
  const { isAuthenticated, isLoading, user, isGuestMode } = useAuth();

  // Debug logs
  console.log('🛡️ ProtectedRoute check:', {
    isLoading,
    isAuthenticated,
    user,
    isGuestMode,
    requiredRole,
    currentPath: window.location.pathname
  });

  if (isLoading) {
    console.log('⏳ Auth is loading, showing spinner...');
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="flex flex-col items-center space-y-4">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-green-600"></div>
          <p className="text-sm text-gray-600">Loading...</p>
        </div>
      </div>
    );
  }

  // Allow access if user is authenticated (includes guest mode)
  if (!isAuthenticated) {
    console.log('❌ User not authenticated, showing access denied...');
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="max-w-md w-full bg-white shadow-lg rounded-lg p-6">
          <div className="text-center">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Access Denied</h2>
            <p className="text-gray-600 mb-4">Please log in to access this page.</p>
            <div className="mb-4 p-3 bg-gray-100 rounded text-sm text-left">
              <strong>Debug Info:</strong><br/>
              isLoading: {isLoading.toString()}<br/>
              isAuthenticated: {isAuthenticated.toString()}<br/>
              isGuestMode: {isGuestMode.toString()}<br/>
              user: {user ? 'exists' : 'null'}<br/>
              token exists: {localStorage.getItem('auth_token') ? 'yes' : 'no'}
            </div>
            <button
              onClick={() => window.location.href = '/login'}
              className="w-full bg-green-600 text-white py-2 px-4 rounded-md hover:bg-green-700 transition-colors"
            >
              Go to Login
            </button>
          </div>
        </div>
      </div>
    );
  }

  // Check role requirements (guest users have 'user' role by default)
  if (requiredRole && user?.role !== requiredRole) {
    console.log('❌ User role mismatch:', user?.role, 'required:', requiredRole);
    
    // If required role is admin but user is guest or regular user
    if (requiredRole === 'admin') {
      return (
        <div className="min-h-screen flex items-center justify-center bg-gray-50">
          <div className="max-w-md w-full bg-white shadow-lg rounded-lg p-6">
            <div className="text-center">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">Access Denied</h2>
              <p className="text-gray-600 mb-4">
                You don't have permission to access this page. Admin access required.
              </p>
              <button
                onClick={() => window.history.back()}
                className="w-full bg-green-600 text-white py-2 px-4 rounded-md hover:bg-green-700 transition-colors"
              >
                Go Back
              </button>
            </div>
          </div>
        </div>
      );
    }
  }

  console.log('✅ Access granted, rendering protected content');
  return <>{children}</>;
}