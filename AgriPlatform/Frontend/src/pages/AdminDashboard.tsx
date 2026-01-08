import { useEffect } from 'react';
import { useState, useRef } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { useFarms } from '../hooks/useFarms';
import { useFarmStore } from '../stores/farmStore';
import { useUserStore } from '../stores/userStore';
import { LogOut, Users, MapPin, Sprout, BarChart3, Settings, Plus, Mail, Phone, Home, Crown, TrendingUp, Activity, Zap } from 'lucide-react';
import { formatHectares } from '@/utils';

export default function AdminDashboard() {
    const [allFarmsPage, setAllFarmsPage] = useState(1);
    const [usersPage, setUsersPage] = useState(1);
    const allFarmsSectionRef = useRef<HTMLDivElement>(null);
    const usersSectionRef = useRef<HTMLDivElement>(null);
    const allFarmsLimit = 10;
    const usersLimit = 10;
    const { user, logout, isAuthenticated } = useAuth();
    
    // Use the unified farms hook for "My Farms"
    const { farms, loading: farmsLoading, error: farmsError, fetchFarms, clearError: clearFarmsError } = useFarms();
    
    // Use farm store directly for system-wide admin functions (all farms)
    const { allFarms, fetchAllFarms, clearError: clearAllFarmsError } = useFarmStore();
    
    const { users, userStats, loading: usersLoading, error: usersError, fetchUsers, fetchUserStats, clearError: clearUsersError } = useUserStore();

    // Use allFarms for system-wide stats, farms for My Farms
    const totalFarms = allFarms.length;
    const totalArea:number = allFarms.reduce((sum, farm) => sum + farm.area, 0);
    const activeCrops = new Set(allFarms.map(farm => farm.crop)).size;
    
    // For admin dashboard, farms from useFarms hook are already the user's own farms
    const myFarms = farms;

    // Debug logs
    console.log('AdminDashboard user:', user);
    console.log('AdminDashboard farms:', farms);
    console.log('AdminDashboard myFarms:', myFarms);

    // Fetch farms and users when component mounts and when user authentication is confirmed
    useEffect(() => {
        if (user && user.id && isAuthenticated) {
            fetchFarms(); // For My Farms - uses unified hook
            fetchAllFarms(allFarmsPage, allFarmsLimit); // For All System Farms, paginated
            fetchUserStats(); // Get user statistics
            fetchUsers(usersPage, usersLimit); // Get users, paginated
        }
    }, [user?.id, isAuthenticated, allFarmsPage, usersPage]);

    // Scroll All System Farms section into view when page changes
    const lastPageRef = useRef(allFarmsPage);
    useEffect(() => {
        if (allFarmsSectionRef.current) {
            if (allFarmsPage > lastPageRef.current) {
                // Next button: scroll to top
                allFarmsSectionRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
            } else if (allFarmsPage < lastPageRef.current) {
                // Previous button: scroll to bottom
                allFarmsSectionRef.current.scrollIntoView({ behavior: 'smooth', block: 'end' });
            }
            lastPageRef.current = allFarmsPage;
        }
    }, [allFarmsPage]);

    // Clear any errors when component unmounts
    useEffect(() => {
        return () => {
            clearFarmsError();
            clearAllFarmsError();
            clearUsersError();
        };
    }, []); // Remove dependencies to avoid infinite loop

    const handleLogout = () => {
        logout();
    };

    if (farmsLoading && usersLoading) {
        return (
            <div className="min-h-screen gradient-mesh flex items-center justify-center">
                <div className="card p-8 flex flex-col items-center space-y-6">
                    <div className="relative">
                        <div className="animate-spin rounded-full h-12 w-12 border-4 border-primary-200 border-t-primary-600"></div>
                        <div className="absolute inset-0 flex items-center justify-center">
                            <Activity className="h-5 w-5 text-primary-600 animate-pulse" />
                        </div>
                    </div>
                    <div className="text-center">
                        <p className="text-lg font-medium text-neutral-900 mb-2">Loading Admin Dashboard</p>
                        <p className="text-sm text-neutral-600">Gathering system insights...</p>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen gradient-mesh">
            {/* Enhanced Header */}
            <header className="glass border-b border-white/10 sticky top-0 z-40">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="flex justify-between items-center py-4">
                        <div className="flex items-center space-x-4 animate-in">
                            <div className="bg-gradient-to-br from-accent-500 to-accent-700 p-2.5 rounded-xl shadow-glow-accent">
                                <Crown className="h-6 w-6 text-white" />
                            </div>
                            <div>
                                <div className="flex items-center space-x-2">
                                    <h1 className="text-2xl font-bold text-neutral-900">Admin Dashboard</h1>
                                    <div className="badge-accent">
                                        <Crown className="h-3 w-3 mr-1" />
                                        Admin
                                    </div>
                                </div>
                                <p className="text-sm text-neutral-600">System Overview - Welcome, {user?.fullName}</p>
                            </div>
                        </div>
                        <div className="flex items-center space-x-3 animate-in stagger-1">
                            <button className="btn-secondary">
                                <Settings className="h-4 w-4 mr-2" />
                                Settings
                            </button>
                            <button
                                onClick={handleLogout}
                                className="btn-ghost"
                            >
                                <LogOut className="h-4 w-4 mr-2" />
                                Sign Out
                            </button>
                        </div>
                    </div>
                </div>
            </header>

            {/* Main Content */}
            <main className="max-w-7xl mx-auto py-8 sm:px-6 lg:px-8">
                <div className="px-4 sm:px-0 space-y-8">
                    {/* Error State */}
                    {(farmsError || usersError) && (
                        <div className="card-elevated bg-red-50 border-l-4 border-l-red-500 p-6 animate-in">
                            <div className="flex items-start space-x-3">
                                <div className="flex-shrink-0">
                                    <Activity className="h-5 w-5 text-red-500" />
                                </div>
                                <div className="flex-1">
                                    <h3 className="text-sm font-semibold text-red-800 mb-2">
                                        Error loading admin data
                                    </h3>
                                    <p className="text-sm text-red-700 mb-4">
                                        {farmsError || usersError}
                                    </p>
                                    <div className="flex space-x-3">
                                        {farmsError && (
                                            <button
                                                onClick={() => fetchFarms()}
                                                className="btn-sm bg-red-100 text-red-800 hover:bg-red-200 border border-red-200"
                                            >
                                                Retry Farms
                                            </button>
                                        )}
                                        {usersError && (
                                            <button
                                                onClick={() => {
                                                    fetchUserStats();
                                                    fetchUsers();
                                                }}
                                                className="btn-sm bg-red-100 text-red-800 hover:bg-red-200 border border-red-200"
                                            >
                                                Retry Users
                                            </button>
                                        )}
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Enhanced System Stats Grid */}
                    <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
                        <div className="card-elevated group  transition-all duration-300 p-6 animate-in">
                            <div className="flex items-center justify-between">
                                <div className="flex-1">
                                    <p className="text-sm font-medium text-neutral-600 mb-1">Total Users</p>
                                    <p className="text-3xl font-bold text-neutral-900">
                                        {userStats ? userStats.totalUsers.toLocaleString() : '...'}
                                    </p>
                                    <div className="flex items-center mt-2">
                                        <TrendingUp className="h-3 w-3 text-primary-600 mr-1" />
                                        <span className="text-xs text-primary-600 font-medium">+12% this month</span>
                                    </div>
                                </div>
                                <div className="flex-shrink-0">
                                    <div className="h-12 w-12 rounded-xl bg-gradient-to-br from-blue-500 to-blue-700 flex items-center justify-center group-hover:shadow-glow transition-all">
                                        <Users className="h-6 w-6 text-white" />
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div className="card-elevated group  transition-all duration-300 p-6 animate-in stagger-1">
                            <div className="flex items-center justify-between">
                                <div className="flex-1">
                                    <p className="text-sm font-medium text-neutral-600 mb-1">Admin Farms</p>
                                    <p className="text-3xl font-bold text-neutral-900">{totalFarms.toLocaleString()}</p>
                                    <div className="flex items-center mt-2">
                                        <TrendingUp className="h-3 w-3 text-primary-600 mr-1" />
                                        <span className="text-xs text-primary-600 font-medium">+8% this month</span>
                                    </div>
                                </div>
                                <div className="flex-shrink-0">
                                    <div className="h-12 w-12 rounded-xl bg-gradient-to-br from-primary-500 to-primary-700 flex items-center justify-center group-hover:shadow-glow transition-all">
                                        <MapPin className="h-6 w-6 text-white" />
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div className="card-elevated group  transition-all duration-300 p-6 animate-in stagger-2">
                            <div className="flex items-center justify-between">
                                <div className="flex-1">
                                    <p className="text-sm font-medium text-neutral-600 mb-1">Total Area</p>
                                    <p className="text-3xl font-bold text-neutral-900">{formatHectares(totalArea)}<span className="text-lg text-neutral-600 ml-1">ha</span></p>
                                    <div className="flex items-center mt-2">
                                        <TrendingUp className="h-3 w-3 text-accent-600 mr-1" />
                                        <span className="text-xs text-accent-600 font-medium">+15% this month</span>
                                    </div>
                                </div>
                                <div className="flex-shrink-0">
                                    <div className="h-12 w-12 rounded-xl bg-gradient-to-br from-accent-500 to-accent-700 flex items-center justify-center group-hover:shadow-glow-accent transition-all">
                                        <BarChart3 className="h-6 w-6 text-white" />
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div className="card-elevated group  transition-all duration-300 p-6 animate-in stagger-3">
                            <div className="flex items-center justify-between">
                                <div className="flex-1">
                                    <p className="text-sm font-medium text-neutral-600 mb-1">Crop Types</p>
                                    <p className="text-3xl font-bold text-neutral-900">{activeCrops}</p>
                                    <div className="flex items-center mt-2">
                                        <Zap className="h-3 w-3 text-secondary-700 mr-1" />
                                        <span className="text-xs text-secondary-700 font-medium">5 new varieties</span>
                                    </div>
                                </div>
                                <div className="flex-shrink-0">
                                    <div className="h-12 w-12 rounded-xl bg-gradient-to-br from-secondary-500 to-secondary-700 flex items-center justify-center group-hover:shadow-glow transition-all">
                                        <Sprout className="h-6 w-6 text-white" />
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    {/* Enhanced Admin's Own Farms */}
                    <div className="animate-in stagger-1">
                        <div className="card-elevated">
                            <div className="p-6">
                                <div className="flex justify-between items-center mb-6">
                                    <div>
                                        <h3 className="text-xl font-semibold text-neutral-900 mb-1">
                                            My Farms
                                        </h3>
                                        <p className="text-sm text-neutral-600">
                                            {myFarms.length} {myFarms.length === 1 ? 'farm' : 'farms'} under your management
                                        </p>
                                    </div>
                                    <Link
                                        to="/create-farm"
                                        className="btn-primary group"
                                    >
                                        <Plus className="h-4 w-4 mr-2 group-hover:rotate-90 transition-transform duration-200" />
                                        Create Farm
                                    </Link>
                                </div>
                                {myFarms.length > 0 ? (
                                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                                        {myFarms.map((farm, index) => (
                                            <div key={farm.id} className={`card group  transition-all duration-300 p-5 border-l-4 border-l-primary-500 slide-in stagger-${Math.min(index + 1, 4)}`}>
                                                <div className="flex justify-between items-start mb-4">
                                                    <div>
                                                        <h4 className="text-lg font-semibold text-neutral-900 mb-1 group-hover:text-primary-600 transition-colors">{farm.name}</h4>
                                                        <div className="flex items-center space-x-2">
                                                            <div className="badge-success">
                                                                <Sprout className="h-3 w-3 mr-1" />
                                                                {farm.crop}
                                                            </div>
                                                        </div>
                                                    </div>
                                                    <div className="flex flex-col items-end space-y-2">
                                                        <div className="h-10 w-10 rounded-lg bg-primary-100 flex items-center justify-center group-hover:bg-primary-200 transition-colors">
                                                            <MapPin className="h-5 w-5 text-primary-600" />
                                                        </div>
                                                        <div className="badge-accent">
                                                            <Crown className="h-3 w-3 mr-1" />
                                                            Admin
                                                        </div>
                                                    </div>
                                                </div>
                                                
                                                <div className="space-y-3">
                                                    <div className="grid grid-cols-2 gap-4 text-sm">
                                                        <div className="bg-neutral-50 rounded-lg p-3">
                                                            <p className="text-neutral-500 text-xs mb-1">Area</p>
                                                            <p className="font-semibold text-neutral-900">{formatHectares(farm.area)} ha</p>
                                                        </div>
                                                        <div className="bg-neutral-50 rounded-lg p-3">
                                                            <p className="text-neutral-500 text-xs mb-1">Status</p>
                                                            <p className="font-semibold text-primary-600">Active</p>
                                                        </div>
                                                    </div>
                                                    
                                                    <div className="space-y-2 text-sm text-neutral-600">
                                                        <div className="flex justify-between items-center">
                                                            <span className="flex items-center">
                                                                <div className="h-2 w-2 bg-primary-500 rounded-full mr-2"></div>
                                                                Planted
                                                            </span>
                                                            <span className="font-medium text-neutral-900">
                                                                {new Date(farm.plantingDate).toLocaleDateString()}
                                                            </span>
                                                        </div>
                                                        <div className="flex justify-between items-center">
                                                            <span className="flex items-center">
                                                                <div className="h-2 w-2 bg-secondary-500 rounded-full mr-2"></div>
                                                                Harvest
                                                            </span>
                                                            <span className="font-medium text-neutral-900">
                                                                {new Date(farm.harvestDate).toLocaleDateString()}
                                                            </span>
                                                        </div>
                                                    </div>
                                                </div>

                                                {farm.description && (
                                                    <div className="mt-4 pt-4 border-t border-neutral-100">
                                                        <p className="text-sm text-neutral-600 line-clamp-2">
                                                            {farm.description}
                                                        </p>
                                                    </div>
                                                )}

                                                <div className="mt-4 pt-4 border-t border-neutral-100 flex justify-between items-center">
                                                    <span className="text-xs text-neutral-400">Created {new Date(farm.createdAt).toLocaleDateString()}</span>
                                                    <Link 
                                                        to={`/farm/${farm.id}`}
                                                        className="btn-sm btn-primary group"
                                                    >
                                                        View Details
                                                        <MapPin className="ml-1 h-3 w-3 group-hover:translate-x-0.5 transition-transform" />
                                                    </Link>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                ) : (
                                    <div className="text-center py-12 bg-gradient-to-br from-neutral-50 to-neutral-100 rounded-2xl border border-dashed border-neutral-200">
                                        <div className="inline-flex items-center justify-center h-16 w-16 rounded-2xl bg-neutral-200 mb-4">
                                            <MapPin className="h-8 w-8 text-neutral-400" />
                                        </div>
                                        <h3 className="text-lg font-medium text-neutral-900 mb-2">No personal farms yet</h3>
                                        <p className="text-sm text-neutral-600 mb-6 max-w-sm mx-auto">
                                            Start your agricultural journey by creating your first farm.
                                        </p>
                                        <Link
                                            to="/create-farm"
                                            className="btn-primary group"
                                        >
                                                <Plus className="h-4 w-4 mr-2" />
                                                Create Farm
                                        </Link>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                    {/* Users Management */}
                    <div className="mt-8" ref={usersSectionRef}>
                        <div className="bg-white shadow rounded-lg">
                            <div className="px-4 py-5 sm:p-6">
                                <div className="flex justify-between items-center mb-4">
                                    <h3 className="text-lg leading-6 font-medium text-gray-900">
                                        User Management
                                    </h3>
                                    <div className="flex items-center space-x-2">
                                        <span className="text-sm text-gray-500">
                                            {userStats && (
                                                <>
                                                    <span className="font-medium text-blue-600">{userStats.totalFarmers}</span> Farmers, 
                                                    <span className="font-medium text-green-600">{userStats.totalAdmins}</span> Admins
                                                </>
                                            )}
                                        </span>
                                    </div>
                                </div>
                                
                                {usersLoading ? (
                                    <div className="text-center py-8">
                                        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
                                        <p className="mt-2 text-sm text-gray-500">Loading users...</p>
                                    </div>
                                ) : users.length > 0 ? (
                                    <>
                                        <div className="overflow-x-auto">
                                            <table className="min-w-full divide-y divide-gray-200">
                                                <thead className="bg-gray-50">
                                                    <tr>
                                                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                                            Name
                                                        </th>
                                                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                                            Email
                                                        </th>
                                                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                                            Role
                                                        </th>
                                                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                                            Contact
                                                        </th>
                                                    </tr>
                                                </thead>
                                                <tbody className="bg-white divide-y divide-gray-200">
                                                    {users.map((user) => (
                                                        <tr key={user._id}>
                                                            <td className="px-6 py-4 whitespace-nowrap">
                                                                <div className="flex items-center">
                                                                    <div className="flex-shrink-0 h-10 w-10 bg-gray-200 rounded-full flex items-center justify-center">
                                                                        <span className="text-gray-500 font-medium">{user.fullName.charAt(0)}</span>
                                                                    </div>
                                                                    <div className="ml-4">
                                                                        <div className="text-sm font-medium text-gray-900">{user.fullName}</div>
                                                                        <div className="text-sm text-gray-500">Joined {new Date(user.createdAt).toLocaleDateString()}</div>
                                                                    </div>
                                                                </div>
                                                            </td>
                                                            <td className="px-6 py-4 whitespace-nowrap">
                                                                <div className="flex items-center">
                                                                    <Mail className="h-4 w-4 text-gray-400 mr-2" />
                                                                    <span className="text-sm text-gray-900">{user.email}</span>
                                                                </div>
                                                            </td>
                                                            <td className="px-6 py-4 whitespace-nowrap">
                                                                <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${user.role === 'admin' ? 'bg-green-100 text-green-800' : 'bg-blue-100 text-blue-800'}`}>
                                                                    {user.role.charAt(0).toUpperCase() + user.role.slice(1)}
                                                                </span>
                                                            </td>
                                                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                                                <div className="flex flex-col space-y-1">
                                                                    <div className="flex items-center">
                                                                        <Phone className="h-4 w-4 text-gray-400 mr-2" />
                                                                        <span>{user.phone}</span>
                                                                    </div>
                                                                    <div className="flex items-center">
                                                                        <Home className="h-4 w-4 text-gray-400 mr-2" />
                                                                        <span className="truncate max-w-xs">{user.address}</span>
                                                                    </div>
                                                                </div>
                                                            </td>
                                                        </tr>
                                                    ))}
                                                </tbody>
                                            </table>
                                        </div>
                                        
                                        {/* Pagination Controls */}
                                        <div className="flex justify-center mt-4">
                                            <button
                                                className="px-4 py-2 mr-2 bg-gray-200 rounded disabled:opacity-50"
                                                disabled={usersPage === 1}
                                                onClick={() => setUsersPage(usersPage - 1)}
                                            >
                                                Previous
                                            </button>
                                            <span className="px-4 py-2">Page {usersPage}</span>
                                            <button
                                                className="px-4 py-2 ml-2 bg-gray-200 rounded disabled:opacity-50"
                                                disabled={users.length < usersLimit}
                                                onClick={() => setUsersPage(usersPage + 1)}
                                            >
                                                Next
                                            </button>
                                        </div>
                                    </>
                                ) : (
                                    <div className="text-center py-8 bg-gray-50 rounded-lg">
                                        <Users className="mx-auto h-12 w-12 text-gray-400" />
                                        <h3 className="mt-2 text-sm font-medium text-gray-900">No users found</h3>
                                        <p className="mt-1 text-sm text-gray-500">
                                            There are no users in the system yet.
                                        </p>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>

                    

                    {/* All System Farms */}
                    <div className="mt-8" ref={allFarmsSectionRef}>
                        <div className="bg-white shadow rounded-lg">
                            <div className="px-4 py-5 sm:p-6">
                                <div className="flex justify-between items-center mb-4">
                                    <h3 className="text-lg leading-6 font-medium text-gray-900">
                                        All System Farms ({allFarms.length} total)
                                    </h3>
                                </div>
                                {allFarms.length > 0 ? (
                                    <>
                                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                                            {allFarms.map((farm) => (
                                                <div key={farm.id} className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
                                                    <div className="flex justify-between items-start mb-3">
                                                        <h4 className="text-md font-medium text-gray-900">{farm.name}</h4>
                                                        <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
                                                            {farm.crop}
                                                        </span>
                                                    </div>
                                                    <div className="space-y-2 text-sm text-gray-600">
                                                        <div className="flex justify-between">
                                                            <span>Owner:</span>
                                                            <span className="font-medium text-blue-600">
                                                                {(
                                                                    farm.userId === user?.id ||
                                                                    (farm.userId && typeof farm.userId === 'object' && '_id' in farm.userId && (farm.userId as any)._id === user?.id)
                                                                ) ? 'Admin' : 'User'}
                                                            </span>
                                                        </div>
                                                        <div className="flex justify-between">
                                                            <span>Area:</span>
                                                            <span className="font-medium">{formatHectares(farm.area)} hectares</span>
                                                        </div>
                                                        <div className="flex justify-between">
                                                            <span>Planting:</span>
                                                            <span className="font-medium">
                                                                {new Date(farm.plantingDate).toLocaleDateString()}
                                                            </span>
                                                        </div>
                                                        <div className="flex justify-between">
                                                            <span>Harvest:</span>
                                                            <span className="font-medium">
                                                                {new Date(farm.harvestDate).toLocaleDateString()}
                                                            </span>
                                                        </div>
                                                    </div>
                                                    {farm.description && (
                                                        <p className="mt-2 text-xs text-gray-500 line-clamp-2">
                                                            {farm.description}
                                                        </p>
                                                    )}
                                                    <div className="mt-3 flex justify-between items-center text-xs">
                                                        <span className="text-gray-400">Created {new Date(farm.createdAt).toLocaleDateString()}</span>
                                                        <Link 
                                                            to={`/farm/${farm.id}`}
                                                            className="text-green-600 hover:text-green-700 font-medium"
                                                        >
                                                            View Details →
                                                        </Link>
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                        {/* Pagination Controls */}
                                        <div className="flex justify-center mt-4">
                                            <button
                                                className="px-4 py-2 mr-2 bg-gray-200 rounded disabled:opacity-50"
                                                disabled={allFarmsPage === 1}
                                                onClick={() => setAllFarmsPage(allFarmsPage - 1)}
                                            >
                                                Previous
                                            </button>
                                            <span className="px-4 py-2">Page {allFarmsPage}</span>
                                            <button
                                                className="px-4 py-2 ml-2 bg-gray-200 rounded disabled:opacity-50"
                                                disabled={allFarms.length < allFarmsLimit}
                                                onClick={() => setAllFarmsPage(allFarmsPage + 1)}
                                            >
                                                Next
                                            </button>
                                        </div>
                                    </>
                                ) : (
                                    <div className="text-center py-8">
                                        <MapPin className="mx-auto h-12 w-12 text-gray-400" />
                                        <h3 className="mt-2 text-sm font-medium text-gray-900">No farms in system yet</h3>
                                        <p className="mt-1 text-sm text-gray-500">
                                            No users have created farms yet.
                                        </p>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                </div>
            </main>
        </div>
    );
}