import { useParams, useNavigate, Link } from 'react-router-dom';
import React from 'react';
import { useAuth } from '../contexts/AuthContext';
import { useFarmStore } from '../stores/farmStore';
import { useFarms } from '../hooks/useFarms';
import { FarmMapView } from '../components/map/FarmMapView';
import { ArrowLeft, MapPin, Calendar, Sprout, Edit, Trash2, Download, FileText, Map, Lock } from 'lucide-react';
import { useEffect } from 'react';
import { formatHectares } from '@/utils';

export default function FarmDetail() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { user, isGuestMode } = useAuth();
  const { getFarmById, deleteFarm, loading, farms } = useFarms();
  const [farm, setFarm] = React.useState<any>(null);


  // Set farm when farms are loaded
  useEffect(() => {
    if (!loading && id && farms.length > 0) {
      setFarm(getFarmById(id));
    }
  }, [loading, id, getFarmById, farms]);

  // Show loader if loading or farms not loaded
  if (loading || !id || farms.length === 0) {
    return (
      <div className="min-h-screen gradient-mesh flex items-center justify-center">
        <div className="card p-8 flex flex-col items-center space-y-6 animate-in">
          <div className="relative">
            <div className="animate-spin rounded-full h-12 w-12 border-4 border-primary-200 border-t-primary-600"></div>
            <div className="absolute inset-0 flex items-center justify-center">
              <Sprout className="h-5 w-5 text-primary-600 animate-pulse" />
            </div>
          </div>
          <div className="text-center">
            <p className="text-lg font-medium text-neutral-900 mb-2">Loading Farm Details</p>
            <p className="text-sm text-neutral-600">Gathering information...</p>
          </div>
        </div>
      </div>
    );
  }

  // Show 'Farm Not Found' only if farms loaded and farm missing
  if (!farm && id && farms.length > 0 && !loading) {
    return (
      <div className="min-h-screen gradient-mesh flex items-center justify-center">
        <div className="card-elevated p-8 text-center max-w-md animate-in">
          <div className="mb-6">
            <div className="h-16 w-16 bg-gradient-to-br from-neutral-400 to-neutral-600 rounded-2xl flex items-center justify-center mx-auto">
              <Sprout className="h-8 w-8 text-white" />
            </div>
          </div>
          <h2 className="text-2xl font-bold text-neutral-900 mb-3">Farm Not Found</h2>
          <p className="text-neutral-600 mb-6 leading-relaxed">
            The farm you're looking for doesn't exist or has been removed.
          </p>
          <Link to="/dashboard" className="btn-primary inline-flex items-center">
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Dashboard
          </Link>
        </div>
      </div>
    );
  }

  console.log("FARM IN FarmDetail.tsx: ", farm);
  
  // Show loading indicator while farms are being fetched
  if (loading || !id || farms.length === 0) {
    return (
      <div className="min-h-screen gradient-mesh flex items-center justify-center">
        <div className="card p-8 flex flex-col items-center space-y-6 animate-in">
          <div className="relative">
            <div className="animate-spin rounded-full h-12 w-12 border-4 border-primary-200 border-t-primary-600"></div>
            <div className="absolute inset-0 flex items-center justify-center">
              <Sprout className="h-5 w-5 text-primary-600 animate-pulse" />
            </div>
          </div>
          <div className="text-center">
            <p className="text-lg font-medium text-neutral-900 mb-2">Loading Farm Details</p>
            <p className="text-sm text-neutral-600">Gathering information...</p>
          </div>
        </div>
      </div>
    );
  }

  // Only show Farm Not Found after farms are loaded and id is present
  if (!farm && id && farms.length > 0 && !loading) {
    return (
      <div className="min-h-screen gradient-mesh flex items-center justify-center">
        <div className="card-elevated p-8 text-center max-w-md animate-in">
          <div className="mb-6">
            <div className="h-16 w-16 bg-gradient-to-br from-neutral-400 to-neutral-600 rounded-2xl flex items-center justify-center mx-auto">
              <Sprout className="h-8 w-8 text-white" />
            </div>
          </div>
          <h2 className="text-2xl font-bold text-neutral-900 mb-3">Farm Not Found</h2>
          <p className="text-neutral-600 mb-6 leading-relaxed">
            The farm you're looking for doesn't exist or has been removed.
          </p>
          <Link to="/dashboard" className="btn-primary inline-flex items-center">
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Dashboard
          </Link>
        </div>
      </div>
    );
  }
  console.log("FARM : ", farm);

  console.log("USER: ", user)

  // Check if user has permission to view this farm
  // Allow guest users to view guest farms
  const isGuestFarm = (farm as any)?.isGuest === true;
  const canView = user?.role === 'admin' || (farm?.userId === user?.id) || isGuestFarm || isGuestMode;
  // Guest users can only edit/delete guest farms, authenticated users can edit their own farms
  const canEdit = user?.role === 'admin' || 
    (farm?.userId === user?.id) || 
    (isGuestFarm && isGuestMode);
  console.log("CAN EDIT : ", canEdit, "CAN View: ", canView)
  if (!canView) {
    return (
      <div className="min-h-screen gradient-mesh flex items-center justify-center">
        <div className="card-elevated p-8 text-center max-w-md animate-in">
          <div className="mb-6">
            <div className="h-16 w-16 bg-gradient-to-br from-red-500 to-red-700 rounded-2xl flex items-center justify-center mx-auto shadow-glow-red">
              <Lock className="h-8 w-8 text-white" />
            </div>
          </div>
          <h2 className="text-2xl font-bold text-neutral-900 mb-3">Access Denied</h2>
          <p className="text-neutral-600 mb-6 leading-relaxed">
            You don't have permission to view this farm.
          </p>
          <Link to="/dashboard" className="btn-primary inline-flex items-center">
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Dashboard
          </Link>
        </div>
      </div>
    );
  }

  const handleDelete = async () => {
    if (confirm(`Are you sure you want to delete "${farm?.name}"? This action cannot be undone.`)) {
      try {
        await deleteFarm(farm?.id || '');
        navigate('/dashboard');
      } catch (error) {
        console.error('Error deleting farm:', error);
        // Error is handled by the store, no need for additional alert
      }
    }
  };

  return (
    <div className="min-h-screen gradient-mesh">
      {/* Enhanced Header */}
      <header className="glass border-b border-white/10 sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div className="flex items-center animate-in">
              <Link
                to="/dashboard"
                className="mr-4 p-2.5 rounded-xl text-neutral-400 hover:text-neutral-600 hover:bg-white/50 transition-all"
              >
                <ArrowLeft className="h-5 w-5" />
              </Link>
              <div className="flex items-start space-x-4">
                <div className="h-12 w-12 bg-gradient-to-br from-primary-500 to-primary-700 rounded-xl flex items-center justify-center shadow-glow">
                  <Sprout className="h-6 w-6 text-white" />
                </div>
                <div>
                  <h1 className="text-3xl font-bold text-neutral-900">{farm?.name}</h1>
                  <div className="flex items-center space-x-3 mt-1">
                    <span className="badge-primary">{farm?.crop}</span>
                    <span className="text-sm text-neutral-600">{formatHectares(farm?.area)} hectares</span>
                  </div>
                </div>
              </div>
            </div>
            {canEdit && (
              <div className="flex items-center space-x-3 animate-in stagger-1">
                <Link
                  to={`/farm/${id}/edit`}
                  className="btn-secondary"
                >
                  <Edit className="h-4 w-4 mr-2" />
                  Edit Farm
                </Link>
                <button
                  onClick={handleDelete}
                  className="btn-danger"
                >
                  <Trash2 className="h-4 w-4 mr-2" />
                  Delete Farm
                </button>
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Enhanced Main Content */}
      <main className="max-w-7xl mx-auto py-8 sm:px-6 lg:px-8">
        <div className="px-4 sm:px-0">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Enhanced Farm Information */}
            <div className="lg:col-span-2 space-y-6">
              <div className="card-elevated animate-in">
                <div className="p-6">
                  <div className="flex items-center space-x-3 mb-6">
                    <div className="h-10 w-10 bg-gradient-to-br from-primary-500 to-primary-700 rounded-xl flex items-center justify-center">
                      <FileText className="h-5 w-5 text-white" />
                    </div>
                    <div>
                      <h3 className="text-xl font-semibold text-neutral-900">Farm Information</h3>
                      <p className="text-sm text-neutral-600">Detailed farm specifications</p>
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="bg-gradient-to-r from-neutral-50 to-neutral-100 rounded-xl p-4">
                      <dt className="text-sm font-medium text-neutral-700 mb-2">Farm Name</dt>
                      <dd className="text-lg font-semibold text-neutral-900">{farm?.name}</dd>
                    </div>
                    
                    <div className="bg-gradient-to-r from-primary-50 to-primary-100 rounded-xl p-4">
                      <dt className="text-sm font-medium text-neutral-700 mb-2">Crop Type</dt>
                      <dd className="flex items-center">
                        <div className="badge-success">
                          <Sprout className="h-4 w-4 mr-1" />
                          {farm?.crop}
                        </div>
                      </dd>
                    </div>
                    
                    <div className="bg-gradient-to-r from-secondary-50 to-secondary-100 rounded-xl p-4">
                      <dt className="text-sm font-medium text-neutral-700 mb-2">Total Area</dt>
                      <dd className="flex items-center text-2xl font-bold text-secondary-700">
                        {formatHectares(farm?.area)} <span className="text-lg text-neutral-600 ml-1">hectares</span>
                      </dd>
                    </div>
                    
                    <div className="bg-gradient-to-r from-accent-50 to-accent-100 rounded-xl p-4">
                      <dt className="text-sm font-medium text-neutral-700 mb-2">Boundary Points</dt>
                      <dd className="flex items-center text-accent-700 font-semibold">
                        <MapPin className="h-4 w-4 mr-2" />
                        {farm?.coordinates.length} mapped points
                      </dd>
                    </div>
                    
                    <div className="bg-gradient-to-r from-blue-50 to-blue-100 rounded-xl p-4">
                      <dt className="text-sm font-medium text-neutral-700 mb-2">Planting Date</dt>
                      <dd className="flex items-center text-blue-700 font-semibold">
                        <Calendar className="h-4 w-4 mr-2" />
                        {new Date(farm?.plantingDate || 0).toLocaleDateString('en-US', {
                          year: 'numeric',
                          month: 'long',
                          day: 'numeric'
                        })}
                      </dd>
                    </div>
                    
                    <div className="bg-gradient-to-r from-green-50 to-green-100 rounded-xl p-4">
                      <dt className="text-sm font-medium text-neutral-700 mb-2">Expected Harvest</dt>
                      <dd className="flex items-center text-green-700 font-semibold">
                        <Calendar className="h-4 w-4 mr-2" />
                        {new Date(farm?.harvestDate || 0).toLocaleDateString('en-US', {
                          year: 'numeric',
                          month: 'long',
                          day: 'numeric'
                        })}
                      </dd>
                    </div>
                  </div>
                  
                  {farm?.description && (
                    <div className="mt-6 p-6 bg-gradient-to-br from-neutral-50 to-neutral-100 rounded-xl border-l-4 border-l-primary-500">
                      <dt className="text-sm font-medium text-neutral-700 mb-3">Farm Description</dt>
                      <dd className="text-neutral-900 leading-relaxed">{farm.description}</dd>
                    </div>
                  )}
                </div>
              </div>

              {/* Enhanced Farm Map View */}
              <div className="card-elevated animate-in stagger-1">
                <div className="p-6">
                  <div className="flex items-center space-x-3 mb-6">
                    <div className="h-10 w-10 bg-gradient-to-br from-accent-500 to-accent-700 rounded-xl flex items-center justify-center">
                      <Map className="h-5 w-5 text-white" />
                    </div>
                    <div>
                      <h3 className="text-xl font-semibold text-neutral-900">Farm Location & Boundary</h3>
                      <p className="text-sm text-neutral-600">Interactive satellite view with boundary mapping</p>
                    </div>
                  </div>
                  <div className="rounded-xl overflow-hidden border border-neutral-200 shadow-soft">
                    <FarmMapView
                      coordinates={farm?.coordinates || [[]]}
                      farmName={farm?.name || ''}
                      height="400px"
                      className="w-full"
                    />
                  </div>
                </div>
              </div>

              {/* Enhanced Coordinates Information */}
              <div className="card-elevated animate-in stagger-2">
                <div className="p-6">
                  <div className="flex items-center space-x-3 mb-6">
                    <div className="h-10 w-10 bg-gradient-to-br from-secondary-500 to-secondary-700 rounded-xl flex items-center justify-center">
                      <MapPin className="h-5 w-5 text-white" />
                    </div>
                    <div>
                      <h3 className="text-xl font-semibold text-neutral-900">Boundary Coordinates</h3>
                      <p className="text-sm text-neutral-600">{farm?.coordinates.length} GPS coordinate points</p>
                    </div>
                  </div>
                  <div className="bg-neutral-50 rounded-xl p-4">
                    <div className="max-h-60 overflow-y-auto">
                      <table className="min-w-full">
                        <thead>
                          <tr className="border-b border-neutral-200">
                            <th className="px-3 py-2 text-left text-xs font-semibold text-neutral-600 uppercase tracking-wider">
                              Point
                            </th>
                            <th className="px-3 py-2 text-left text-xs font-semibold text-neutral-600 uppercase tracking-wider">
                              Longitude
                            </th>
                            <th className="px-3 py-2 text-left text-xs font-semibold text-neutral-600 uppercase tracking-wider">
                              Latitude
                            </th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-neutral-200">
                          {Array.isArray(farm?.coordinates) &&
                            (farm.coordinates as number[][]).map((coord: number[], index: number) => (
                              Array.isArray(coord) ? (
                                <tr key={index} className="hover:bg-neutral-100 transition-colors">
                                  <td className="px-3 py-2 whitespace-nowrap text-sm font-medium text-neutral-900">
                                    #{index + 1}
                                  </td>
                                  <td className="px-3 py-2 whitespace-nowrap text-sm text-neutral-700 font-mono">
                                    {typeof coord[0] === 'number' ? coord[0].toFixed(6) : ''}
                                  </td>
                                  <td className="px-3 py-2 whitespace-nowrap text-sm text-neutral-700 font-mono">
                                    {typeof coord[1] === 'number' ? coord[1].toFixed(6) : ''}
                                  </td>
                                </tr>
                              ) : null
                            ))
                          }
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Enhanced Sidebar */}
            <div className="lg:col-span-1 space-y-6">
              {/* Enhanced Farm Status */}
              <div className="card-elevated animate-in stagger-3">
                <div className="p-6">
                  <div className="flex items-center space-x-3 mb-6">
                    <div className="h-10 w-10 bg-gradient-to-br from-green-500 to-green-700 rounded-xl flex items-center justify-center">
                      <Sprout className="h-5 w-5 text-white" />
                    </div>
                    <div>
                      <h3 className="text-xl font-semibold text-neutral-900">Farm Status</h3>
                      <p className="text-sm text-neutral-600">Current cultivation status</p>
                    </div>
                  </div>
                  <div className="space-y-4">
                    <div className="bg-gradient-to-r from-green-50 to-green-100 rounded-xl p-4">
                      <dt className="text-sm font-medium text-neutral-700 mb-2">Current Status</dt>
                      <dd className="flex items-center">
                        <div className="badge-success">
                          <div className="h-2 w-2 bg-green-500 rounded-full mr-2 animate-pulse"></div>
                          Active & Growing
                        </div>
                      </dd>
                    </div>
                    
                    <div className="bg-gradient-to-r from-orange-50 to-orange-100 rounded-xl p-4">
                      <dt className="text-sm font-medium text-neutral-700 mb-2">Days to Harvest</dt>
                      <dd className="text-lg font-bold text-orange-700">
                        {(() => {
                          const daysToHarvest = Math.ceil((new Date(farm?.harvestDate|| 0).getTime() - new Date().getTime()) / (1000 * 60 * 60 * 24));
                          if (daysToHarvest < 0) {
                            return (
                              <span className="text-green-700">
                                Harvested {Math.abs(daysToHarvest)} days ago ✓
                              </span>
                            );
                          } else if (daysToHarvest === 0) {
                            return <span className="text-red-600">Harvest today! 🌾</span>;
                          } else {
                            return `${daysToHarvest} days remaining`;
                          }
                        })()}
                      </dd>
                    </div>
                    
                    <div className="bg-gradient-to-r from-blue-50 to-blue-100 rounded-xl p-4">
                      <dt className="text-sm font-medium text-neutral-700 mb-2">Growing Period</dt>
                      <dd className="text-lg font-semibold text-blue-700">
                        {Math.ceil((new Date(farm?.harvestDate || 0).getTime() - new Date(farm?.plantingDate || 0).getTime()) / (1000 * 60 * 60 * 24))} days
                        <span className="text-sm text-blue-600 block">cultivation cycle</span>
                      </dd>
                    </div>
                    
                    <div className="grid grid-cols-1 gap-3">
                      <div className="bg-gradient-to-r from-purple-50 to-purple-100 rounded-xl p-3">
                        <dt className="text-xs font-medium text-neutral-700 mb-1">Created</dt>
                        <dd className="text-sm font-semibold text-purple-700">
                          {new Date(farm?.createdAt || 0).toLocaleDateString('en-US', {
                            year: 'numeric',
                            month: 'short',
                            day: 'numeric'
                          })}
                        </dd>
                      </div>
                      
                      <div className="bg-gradient-to-r from-indigo-50 to-indigo-100 rounded-xl p-3">
                        <dt className="text-xs font-medium text-neutral-700 mb-1">Last Updated</dt>
                        <dd className="text-sm font-semibold text-indigo-700">
                          {new Date(farm?.updatedAt || 0).toLocaleDateString('en-US', {
                            year: 'numeric',
                            month: 'short',
                            day: 'numeric'
                          })}
                        </dd>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Enhanced Quick Actions */}
              <div className="card-elevated animate-in stagger-4">
                <div className="p-6">
                  <div className="flex items-center space-x-3 mb-6">
                    <div className="h-10 w-10 bg-gradient-to-br from-accent-500 to-accent-700 rounded-xl flex items-center justify-center">
                      <Sprout className="h-5 w-5 text-white" />
                    </div>
                    <div>
                      <h3 className="text-xl font-semibold text-neutral-900">Quick Actions</h3>
                      <p className="text-sm text-neutral-600">Manage farm operations</p>
                    </div>
                  </div>
                  <div className="space-y-3">
                    <button 
                      onClick={() => {
                        const mapElement = document.querySelector('.leaflet-container');
                        if (mapElement) {
                          mapElement.scrollIntoView({ behavior: 'smooth' });
                        }
                      }}
                      className="w-full group relative overflow-hidden bg-gradient-to-r from-blue-50 to-blue-100 hover:from-blue-100 hover:to-blue-200 border border-blue-200 text-blue-800 rounded-xl px-4 py-4 text-sm font-medium transition-all duration-200 hover:shadow-soft"
                    >
                      <div className="flex items-center">
                        <div className="h-8 w-8 bg-blue-500 rounded-lg flex items-center justify-center mr-3 group-hover:scale-110 transition-transform">
                          <Map className="h-4 w-4 text-white" />
                        </div>
                        <div className="text-left">
                          <div className="font-semibold">View on Map</div>
                          <div className="text-xs text-blue-600">Scroll to interactive map</div>
                        </div>
                      </div>
                    </button>
                    
                    <button 
                      onClick={() => {
                        const reportData = {
                          farmName: farm?.name || '',
                          crop: farm?.crop || '',
                          area: farm?.area || 0,
                          plantingDate: farm?.plantingDate || '',
                          harvestDate: farm?.harvestDate || '',
                          coordinates: farm?.coordinates || [],
                          createdAt: farm?.createdAt || ''
                        };
                        console.log('Generating report for:', reportData);
                        alert('Report generation feature coming soon!');
                      }}
                      className="w-full group relative overflow-hidden bg-gradient-to-r from-purple-50 to-purple-100 hover:from-purple-100 hover:to-purple-200 border border-purple-200 text-purple-800 rounded-xl px-4 py-4 text-sm font-medium transition-all duration-200 hover:shadow-soft"
                    >
                      <div className="flex items-center">
                        <div className="h-8 w-8 bg-purple-500 rounded-lg flex items-center justify-center mr-3 group-hover:scale-110 transition-transform">
                          <FileText className="h-4 w-4 text-white" />
                        </div>
                        <div className="text-left">
                          <div className="font-semibold">Generate Report</div>
                          <div className="text-xs text-purple-600">Farm analysis & insights</div>
                        </div>
                      </div>
                    </button>
                    
                    <button 
                      onClick={() => {
                        const farmData = {
                          id: farm?.id || '',
                          name: farm?.name || '',
                          crop: farm?.crop || '',
                          area: farm?.area || 0,
                          plantingDate: farm?.plantingDate || '',
                          harvestDate: farm?.harvestDate || '',
                          description: farm?.description || '',
                          coordinates: farm?.coordinates || [],
                          createdAt: farm?.createdAt || '',
                          updatedAt: farm?.updatedAt || ''
                        };
                        
                        const dataStr = JSON.stringify(farmData, null, 2);
                        const dataBlob = new Blob([dataStr], { type: 'application/json' });
                        const url = URL.createObjectURL(dataBlob);
                        const link = document.createElement('a');
                        link.href = url;
                        link.download = `${farm?.name.replace(/\s+/g, '_')}_farm_data.json`;
                        document.body.appendChild(link);
                        link.click();
                        document.body.removeChild(link);
                        URL.revokeObjectURL(url);
                      }}
                      className="w-full group relative overflow-hidden bg-gradient-to-r from-green-50 to-green-100 hover:from-green-100 hover:to-green-200 border border-green-200 text-green-800 rounded-xl px-4 py-4 text-sm font-medium transition-all duration-200 hover:shadow-soft"
                    >
                      <div className="flex items-center">
                        <div className="h-8 w-8 bg-green-500 rounded-lg flex items-center justify-center mr-3 group-hover:scale-110 transition-transform">
                          <Download className="h-4 w-4 text-white" />
                        </div>
                        <div className="text-left">
                          <div className="font-semibold">Export Data</div>
                          <div className="text-xs text-green-600">Download JSON file</div>
                        </div>
                      </div>
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}