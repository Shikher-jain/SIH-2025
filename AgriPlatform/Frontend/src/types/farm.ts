export interface Farm {
  id: string;
  name: string;
  crop: string;
  plantingDate: string;
  harvestDate: string;
  description?: string;
  coordinates: number[][]; // Array of [lng, lat] pairs for polygon
  area: number; // in hectares
  createdAt: string;
  updatedAt: string;
  userId: string;
}

export interface FarmFormData {
  name: string;
  crop: string;
  plantingDate: string;
  harvestDate: string;
  description?: string;
}

export interface FarmState {
  farms: Farm[];
  allFarms: Farm[];
  currentFarm: Farm | null;
  loading: boolean;
  error: string | null;
  guestMode: boolean;
  pagination: {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
  };
  fetchFarms: (page?: number, limit?: number) => Promise<void>;
  fetchAllFarms: (page?: number, limit?: number) => Promise<void>;
  addFarm: (
    farmData: FarmFormData,
    coordinates: number[][],
    area: number
  ) => Promise<void>;
  updateFarm: (id: string, farmData: Partial<Farm>) => Promise<void>;
  deleteFarm: (id: string) => Promise<void>;
  getFarmById: (id: string) => Farm | undefined;
  setCurrentFarm: (farm: Farm | null) => void;
  clearError: () => void;
  clearUserData: () => void;
  setPagination: (pagination: Partial<FarmState['pagination']>) => void;
  setGuestMode: (isGuest: boolean) => void;
  clearAllData: () => void;
}

export interface FarmStore extends FarmState {
  fetchAllFarms: (page?: number, limit?: number) => Promise<void>;
  fetchFarms: (page?: number, limit?: number) => Promise<void>;
  addFarm: (
    farmData: FarmFormData,
    coordinates: number[][],
    area: number
  ) => Promise<void>;
  updateFarm: (id: string, farmData: Partial<Farm>) => Promise<void>;
  deleteFarm: (id: string) => Promise<void>;
  getFarmById: (id: string) => Farm | undefined;
  setCurrentFarm: (farm: Farm | null) => void;
  clearError: () => void;
  clearUserData: () => void;
  setPagination: (pagination: Partial<FarmState['pagination']>) => void;
  setGuestMode: (isGuest: boolean) => void;
  clearAllData: () => void;
}

export const CROP_OPTIONS = [
  'Wheat',
  'Rice',
  'Corn',
  'Soybeans',
  'Cotton',
  'Sugarcane',
  'Potato',
  'Tomato',
  'Onion',
  'Cabbage',
  'Carrot',
  'Beans',
  'Peas',
  'Sunflower',
  'Barley',
  'Oats',
] as const;

export type CropType = (typeof CROP_OPTIONS)[number];
