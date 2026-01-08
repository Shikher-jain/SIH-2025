import { useEffect, useCallback } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { useFarmStore } from '../stores/farmStore';
import { useGuestFarmStore } from '../stores/guestFarmStore';
import type { Farm, FarmFormData } from '../types/farm';

/**
 * Unified hook for farm operations that automatically switches between
 * authenticated user farms (API-based) and guest farms (localStorage-based)
 */
export const useFarms = () => {
  const { user, isGuestMode, isAuthenticated } = useAuth();

  // API-based farm store for authenticated users
  const {
    farms: apiFarms,
    currentFarm: apiCurrentFarm,
    loading: apiLoading,
    error: apiError,
    fetchFarms: apiFetchFarms,
    addFarm: apiAddFarm,
    updateFarm: apiUpdateFarm,
    deleteFarm: apiDeleteFarm,
    getFarmById: apiGetFarmById,
    setCurrentFarm: apiSetCurrentFarm,
    clearError: apiClearError,
    clearUserData: apiClearUserData,
  } = useFarmStore();

  // Guest farm store for guest mode
  const {
    farms: guestFarms,
    currentFarm: guestCurrentFarm,
    loading: guestLoading,
    error: guestError,
    fetchGuestFarms,
    addGuestFarm,
    updateGuestFarm,
    deleteGuestFarm,
    getFarmById: guestGetFarmById,
    setCurrentFarm: guestSetCurrentFarm,
    clearError: guestClearError,
    clearGuestData,
  } = useGuestFarmStore();

  // Determine which store to use
  // Use guest mode if isGuestMode is true and user is NOT authenticated
  const isUsingGuestMode = isGuestMode;

  // Unified state
  const farms = isUsingGuestMode ? guestFarms : apiFarms;
  const currentFarm = isUsingGuestMode ? guestCurrentFarm : apiCurrentFarm;
  const loading = isUsingGuestMode ? guestLoading : apiLoading;
  const error = isUsingGuestMode ? guestError : apiError;

  // Auto-fetch farms when user context is set (after refresh/login)
  useEffect(() => {
    if (user && user.id) {
      if (isUsingGuestMode) {
        fetchGuestFarms();
      } else {
        apiFetchFarms();
      }
    }
  }, [user?.id, isUsingGuestMode]);

  // Unified methods
  const fetchFarms = useCallback(async () => {
    if (isUsingGuestMode) {
      fetchGuestFarms();
    } else {
      await apiFetchFarms();
    }
  }, [isUsingGuestMode, fetchGuestFarms, apiFetchFarms]);

  const addFarm = useCallback(
    async (farmData: FarmFormData, coordinates: number[][], area: number) => {
      if (isUsingGuestMode) {
        addGuestFarm(farmData, coordinates, area);
      } else {
        await apiAddFarm(farmData, coordinates, area);
      }
    },
    [isUsingGuestMode, addGuestFarm, apiAddFarm]
  );

  const updateFarm = useCallback(
    async (id: string, farmData: Partial<Farm>) => {
      if (isUsingGuestMode) {
        updateGuestFarm(id, farmData);
      } else {
        await apiUpdateFarm(id, farmData);
      }
    },
    [isUsingGuestMode, updateGuestFarm, apiUpdateFarm]
  );

  const deleteFarm = useCallback(
    async (id: string) => {
      if (isUsingGuestMode) {
        deleteGuestFarm(id);
      } else {
        await apiDeleteFarm(id);
      }
    },
    [isUsingGuestMode, deleteGuestFarm, apiDeleteFarm]
  );

  const getFarmById = useCallback(
    (id: string): Farm | undefined => {
      if (isUsingGuestMode) {
        return guestGetFarmById(id);
      } else {
        return apiGetFarmById(id);
      }
    },
    [isUsingGuestMode, guestGetFarmById, apiGetFarmById]
  );

  const setCurrentFarm = useCallback(
    (farm: Farm | null) => {
      if (isUsingGuestMode) {
        guestSetCurrentFarm(farm);
      } else {
        apiSetCurrentFarm(farm);
      }
    },
    [isUsingGuestMode, guestSetCurrentFarm, apiSetCurrentFarm]
  );

  const clearError = useCallback(() => {
    if (isUsingGuestMode) {
      guestClearError();
    } else {
      apiClearError();
    }
  }, [isUsingGuestMode, guestClearError, apiClearError]);

  const clearAllData = useCallback(() => {
    if (isUsingGuestMode) {
      clearGuestData();
    } else {
      apiClearUserData();
    }
  }, [isUsingGuestMode, clearGuestData, apiClearUserData]);

  return {
    // State
    farms,
    currentFarm,
    loading,
    error,
    isGuestMode: isUsingGuestMode,

    // Methods
    fetchFarms,
    addFarm,
    updateFarm,
    deleteFarm,
    getFarmById,
    setCurrentFarm,
    clearError,
    clearAllData,
  };
};
