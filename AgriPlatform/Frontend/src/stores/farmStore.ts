import { create } from 'zustand';
import { FarmAPI } from '../services/farmApi';
import type { Farm, FarmState, FarmFormData } from '../types/farm';
import {
  loadGuestFarms,
  calculatePolygonArea,
  GuestFarmStorage,
} from '../utils/guestFarmStorage';

export const useFarmStore = create<FarmState>((set, get) => ({
  fetchAllFarms: async (page = 1, limit = 10) => {
    set({ loading: true, error: null });
    try {
      const response = await FarmAPI.getAllFarms(page, limit);
      if (response.code === 1) {
        const { farms, pagination } = response.result;
        const transformedFarms = farms.map(FarmAPI.transformFromApiFormat);
        set({ allFarms: transformedFarms, pagination, loading: false });
      } else {
        set({
          error: response.message || 'Failed to fetch all farms',
          loading: false,
        });
      }
    } catch (error: any) {
      console.error('Error fetching all farms:', error);
      set({
        error: error.message || 'Failed to fetch all farms',
        loading: false,
      });
    }
  },
  guestMode:
    typeof window !== 'undefined' && !localStorage.getItem('auth_token'),
  farms: [],
  allFarms: [],
  currentFarm: null,
  loading: false,
  error: null,
  pagination: {
    page: 1,
    limit: 10,
    total: 0,
    totalPages: 0,
  },
  fetchFarms: async (page = 1, limit = 10) => {
    // Check current authentication status dynamically
    const isAuthenticated = !!(
      typeof window !== 'undefined' &&
      localStorage.getItem('auth_token') &&
      localStorage.getItem('auth_user')
    );

    // Use guest mode if not authenticated
    if (!isAuthenticated || get().guestMode) {
      set({ loading: true, error: null, guestMode: true });
      const guestFarms = loadGuestFarms().map(farm => ({
        ...farm,
        userId: 'guest',
      }));
      set({
        farms: guestFarms,
        loading: false,
        pagination: {
          page: 1,
          limit: 100,
          total: guestFarms.length,
          totalPages: 1,
        },
      });
      return;
    }

    try {
      set({ loading: true, error: null, guestMode: false });
      // Always fetch current user's farms for My Farms (admin and normal users)
      const response = await FarmAPI.getFarms(page, limit);
      if (response.code === 1) {
        const { farms, pagination } = response.result;
        const transformedFarms = farms.map(FarmAPI.transformFromApiFormat);
        set({ farms: transformedFarms, pagination, loading: false });
      } else {
        set({
          error: response.message || 'Failed to fetch farms',
          loading: false,
        });
      }
    } catch (error: any) {
      console.error('Error fetching farms:', error);
      set({ error: error.message || 'Failed to fetch farms', loading: false });
    }
  },
  addFarm: async (
    farmData: FarmFormData,
    coordinates: number[][],
    area: number
  ) => {
    if (get().guestMode) {
      set({ loading: true, error: null });
      GuestFarmStorage.addFarm(
        farmData,
        coordinates,
        area || calculatePolygonArea(coordinates)
      );
      const guestFarms = loadGuestFarms().map(farm => ({
        ...farm,
        userId: 'guest',
      }));
      set({ farms: guestFarms, loading: false });
      return;
    }
    try {
      set({ loading: true, error: null });
      const createRequest = FarmAPI.transformToApiFormat({
        ...farmData,
        coordinates,
        area: area || calculatePolygonArea(coordinates),
      });
      const response = await FarmAPI.createFarm(createRequest);
      if (response.code === 1) {
        // Force refetch farms from backend after creation
        await get().fetchFarms();
        // Also refetch all farms for admin dashboard
        if (typeof get().fetchAllFarms === 'function') {
          await get().fetchAllFarms();
        }
        set({ loading: false });
      } else {
        set({
          error: response.message || 'Failed to create farm',
          loading: false,
        });
      }
    } catch (error: any) {
      console.error('Error creating farm:', error);
      set({ error: error.message || 'Failed to create farm', loading: false });
    }
  },
  updateFarm: async (id: string, farmData: Partial<Farm>) => {
    if (get().guestMode) {
      set({ loading: true, error: null });
      GuestFarmStorage.updateFarm(id, farmData);
      const guestFarms = loadGuestFarms().map(farm => ({
        ...farm,
        userId: 'guest',
      }));
      set({
        farms: guestFarms,
        currentFarm: guestFarms.find(f => f.id === id) || null,
        loading: false,
      });
      return;
    }
    try {
      set({ loading: true, error: null });
      const response = await FarmAPI.updateFarm(id, farmData);
      if (response.code === 1) {
        const updatedFarm = FarmAPI.transformFromApiFormat(response.result);
        set({
          farms: get().farms.map((farm: Farm) =>
            farm.id === id ? updatedFarm : farm
          ),
          currentFarm:
            get().currentFarm?.id === id ? updatedFarm : get().currentFarm,
          loading: false,
        });
      } else {
        set({
          error: response.message || 'Failed to update farm',
          loading: false,
        });
      }
    } catch (error: any) {
      console.error('Error updating farm:', error);
      set({ error: error.message || 'Failed to update farm', loading: false });
    }
  },
  deleteFarm: async (id: string) => {
    if (get().guestMode) {
      set({ loading: true, error: null });
      GuestFarmStorage.deleteFarm(id);
      const guestFarms = loadGuestFarms().map(farm => ({
        ...farm,
        userId: 'guest',
      }));
      set({
        farms: guestFarms,
        currentFarm: get().currentFarm?.id === id ? null : get().currentFarm,
        loading: false,
      });
      return;
    }
    try {
      set({ loading: true, error: null });
      const response = await FarmAPI.deleteFarm(id);
      if (response.code === 1) {
        set({
          farms: get().farms.filter((farm: Farm) => farm.id !== id),
          currentFarm: get().currentFarm?.id === id ? null : get().currentFarm,
          loading: false,
        });
      } else {
        set({
          error: response.message || 'Failed to delete farm',
          loading: false,
        });
      }
    } catch (error: any) {
      console.error('Error deleting farm:', error);
      set({ error: error.message || 'Failed to delete farm', loading: false });
    }
  },
  getFarmById: (id: string) => {
    const userFarm = get().farms.find((farm: Farm) => farm.id === id);
    if (userFarm) return userFarm;
    return get().allFarms.find((farm: Farm) => farm.id === id);
  },
  setCurrentFarm: (farm: Farm | null) => {
    set({ currentFarm: farm });
  },
  clearError: () => {
    set({ error: null });
  },
  clearUserData: () => {
    set({
      farms: [],
      currentFarm: null,
      loading: false,
      error: null,
      pagination: { page: 1, limit: 10, total: 0, totalPages: 0 },
      guestMode:
        typeof window !== 'undefined' && !localStorage.getItem('auth_token'),
    });
  },
  setPagination: (pagination: Partial<FarmState['pagination']>) => {
    set({ pagination: { ...get().pagination, ...pagination } });
  },
  setGuestMode: (isGuest: boolean) => {
    set({ guestMode: isGuest });
  },
  clearAllData: () => {
    set({
      farms: [],
      allFarms: [],
      currentFarm: null,
      loading: false,
      error: null,
      pagination: { page: 1, limit: 10, total: 0, totalPages: 0 },
      guestMode:
        typeof window !== 'undefined' && !localStorage.getItem('auth_token'),
    });
  },
}));
