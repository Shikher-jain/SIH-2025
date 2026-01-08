import { create } from 'zustand';
import { GuestFarmStorage, type GuestFarm } from '../utils/guestFarmStorage';
import type { Farm, FarmState, FarmFormData } from '../types/farm';

interface GuestFarmStore extends FarmState {
  // Guest-specific CRUD operations
  addGuestFarm: (
    farmData: FarmFormData,
    coordinates: number[][],
    area: number
  ) => void;
  updateGuestFarm: (id: string, farmData: Partial<FarmFormData & { coordinates?: number[][]; area?: number }>) => void;
  deleteGuestFarm: (id: string) => void;
  fetchGuestFarms: () => void;
  
  // Local state management
  getFarmById: (id: string) => Farm | undefined;
  setCurrentFarm: (farm: Farm | null) => void;
  clearError: () => void;
  clearGuestData: () => void;
}

// Utility function to convert GuestFarm to Farm for consistency
const convertGuestFarmToFarm = (guestFarm: GuestFarm): Farm => ({
  ...guestFarm,
  userId: 'guest', // Ensure userId is consistent
});

export const useGuestFarmStore = create<GuestFarmStore>((set, get) => ({
  // State
  farms: [],
  currentFarm: null,
  loading: false,
  error: null,

  // Fetch guest farms from localStorage
  fetchGuestFarms: () => {
    try {
      set({ loading: true, error: null });
      const guestFarms = GuestFarmStorage.getFarms();
      const farms = guestFarms.map(convertGuestFarmToFarm);
      
      set({
        farms,
        loading: false,
      });
      
      console.log(`📱 Loaded ${farms.length} guest farms from localStorage`);
    } catch (error: any) {
      console.error('Error fetching guest farms:', error);
      set({
        error: error.message || 'Failed to fetch guest farms',
        loading: false,
      });
    }
  },

  addGuestFarm: (
    farmData: FarmFormData,
    coordinates: number[][],
    area: number
  ) => {
    try {
      set({ loading: true, error: null });
      
      const guestFarm = GuestFarmStorage.addFarm(farmData, coordinates, area);
      const newFarm = convertGuestFarmToFarm(guestFarm);
      
      set(state => ({
        farms: [...state.farms, newFarm],
        loading: false,
      }));
      
      console.log('✅ Added guest farm:', newFarm.name);
    } catch (error: any) {
      console.error('Error adding guest farm:', error);
      set({
        error: error.message || 'Failed to create farm',
        loading: false,
      });
    }
  },

  updateGuestFarm: (id: string, farmData: Partial<FarmFormData & { coordinates?: number[][]; area?: number }>) => {
    try {
      set({ loading: true, error: null });
      
      const updatedGuestFarm = GuestFarmStorage.updateFarm(id, farmData);
      
      if (updatedGuestFarm) {
        const updatedFarm = convertGuestFarmToFarm(updatedGuestFarm);
        
        set(state => ({
          farms: state.farms.map(farm =>
            farm.id === id ? updatedFarm : farm
          ),
          currentFarm: state.currentFarm?.id === id ? updatedFarm : state.currentFarm,
          loading: false,
        }));
        
        console.log('✅ Updated guest farm:', updatedFarm.name);
      } else {
        set({
          error: 'Farm not found',
          loading: false,
        });
      }
    } catch (error: any) {
      console.error('Error updating guest farm:', error);
      set({
        error: error.message || 'Failed to update farm',
        loading: false,
      });
    }
  },

  deleteGuestFarm: (id: string) => {
    try {
      set({ loading: true, error: null });
      
      const success = GuestFarmStorage.deleteFarm(id);
      
      if (success) {
        set(state => ({
          farms: state.farms.filter(farm => farm.id !== id),
          currentFarm: state.currentFarm?.id === id ? null : state.currentFarm,
          loading: false,
        }));
        
        console.log('✅ Deleted guest farm:', id);
      } else {
        set({
          error: 'Farm not found',
          loading: false,
        });
      }
    } catch (error: any) {
      console.error('Error deleting guest farm:', error);
      set({
        error: error.message || 'Failed to delete farm',
        loading: false,
      });
    }
  },

  // Local state management
  getFarmById: (id: string) => {
    return get().farms.find(farm => farm.id === id);
  },

  setCurrentFarm: (farm: Farm | null) => {
    set({ currentFarm: farm });
  },

  clearError: () => {
    set({ error: null });
  },

  clearGuestData: () => {
    GuestFarmStorage.clearAllFarms();
    set({
      farms: [],
      currentFarm: null,
      loading: false,
      error: null,
    });
  },
}));