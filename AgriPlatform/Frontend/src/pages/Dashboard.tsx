
import { useAuth } from '../contexts/AuthContext';
import { useEffect } from 'react';
import { useFarmStore } from '../stores/farmStore';
import AdminDashboard from './AdminDashboard';
import UserDashboard from './DashboardPage';

export default function Dashboard() {
  const { user, isGuestMode } = useAuth();
  const { setGuestMode, clearUserData } = useFarmStore();

  useEffect(() => {
    // Sync farm store guest mode with auth context
    setGuestMode(!!isGuestMode);
    
    // Clear user-specific data when switching to guest mode
    // Keep allFarms for admin functionality
    if (isGuestMode) {
      clearUserData();
    }
  }, [isGuestMode, setGuestMode, clearUserData]);

  // Route to appropriate dashboard based on user role
  if (user?.role === 'admin') {
    return <AdminDashboard />;
  }
  return <UserDashboard />;
}