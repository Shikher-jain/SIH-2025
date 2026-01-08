import { get, post, put, del } from '../utils/api';
import { AuthAPI } from './authApi';
import type { Farm } from '../types/farm';

// API Request/Response Types
export interface CreateFarmRequest {
  name: string;
  crop: string;
  plantingDate: string;
  harvestDate: string;
  description?: string;
  coordinates: number[][];
  area: number;
}

export interface UpdateFarmRequest extends Partial<CreateFarmRequest> {}

export interface FarmResponse {
  code: number;
  message: string;
  result:
    | Farm
    | Farm[]
    | {
        farms: Farm[];
        pagination: {
          page: number;
          limit: number;
          total: number;
          totalPages: number;
        };
      };
}

export interface PaginatedFarmsResponse {
  code: number;
  message: string;
  result: {
    farms: Farm[];
    pagination: {
      page: number;
      limit: number;
      total: number;
      totalPages: number;
    };
  };
}

/**
 * Farm API Service
 */
export class FarmAPI {
  /**
   * Get all farms in the system (admin)
   */
  static async getAllFarms(
    page = 1,
    limit = 10
  ): Promise<PaginatedFarmsResponse> {
    try {
      const authConfig = AuthAPI.getAuthConfig();
      const response = await get<PaginatedFarmsResponse>(
        `/farms/all?page=${page}&limit=${limit}`,
        authConfig
      );
      return response;
    } catch (error) {
      console.error('Error fetching all farms:', error);
      throw error;
    }
  }
  /**
   * Get all farms for authenticated user
   */
  static async getFarms(page = 1, limit = 10): Promise<PaginatedFarmsResponse> {
    try {
      const authConfig = AuthAPI.getAuthConfig();
      const response = await get<PaginatedFarmsResponse>(
        `/farms?page=${page}&limit=${limit}`,
        authConfig
      );
      return response;
    } catch (error) {
      console.error('Error fetching farms:', error);
      throw error;
    }
  }

  /**
   * Get single farm by ID
   */
  static async getFarm(id: string): Promise<FarmResponse> {
    try {
      const authConfig = AuthAPI.getAuthConfig();
      const response = await get<FarmResponse>(`/farms/${id}`, authConfig);
      return response;
    } catch (error) {
      console.error('Error fetching farm:', error);
      throw error;
    }
  }

  /**
   * Create new farm
   */
  static async createFarm(farmData: CreateFarmRequest): Promise<FarmResponse> {
    try {
      const authConfig = {
        ...AuthAPI.getAuthConfig(),
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${AuthAPI.getToken()}`, // agar tum token use kar rahe ho
        },
        withCredentials: true,
      };

      const response = await post<FarmResponse>('/farms', farmData, authConfig);
      return response;
    } catch (error) {
      console.error('Error creating farm:', error);
      throw error;
    }
  }

  /**
   * Update existing farm
   */
  /**
   * Update existing farm
   */
  static async updateFarm(
    id: string,
    farmData: UpdateFarmRequest
  ): Promise<FarmResponse> {
    try {
      const authConfig = {
        ...AuthAPI.getAuthConfig(),
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${AuthAPI.getToken()}`,
        },
        withCredentials: true,
      };

      // Format coordinates properly for the backend if they exist
      let formattedData = { ...farmData };

      if (formattedData.coordinates && formattedData.coordinates.length > 0) {
        // Ensure the polygon is closed (first point equals last point)
        const coords = [...formattedData.coordinates];

        // Check if the polygon is not already closed
        const firstPoint = coords[0];
        const lastPoint = coords[coords.length - 1];

        if (firstPoint[0] !== lastPoint[0] || firstPoint[1] !== lastPoint[1]) {
          // Add the first point again to close the loop
          coords.push([...firstPoint]);
        }

        formattedData = {
          ...formattedData,
          coordinates: {
            type: 'Polygon',
            coordinates: [coords],
          },
        };
      }

      const response = await put<FarmResponse>(
        `/farms/${id}`,
        formattedData,
        authConfig
      );
      return response;
    } catch (error) {
      console.error('Error updating farm:', error);
      throw error;
    }
  }

  /**
   * Delete farm
   */
  static async deleteFarm(id: string): Promise<FarmResponse> {
    try {
      const authConfig = AuthAPI.getAuthConfig();
      const response = await del<FarmResponse>(`/farms/${id}`, authConfig);
      return response;
    } catch (error) {
      console.error('Error deleting farm:', error);
      throw error;
    }
  }

  /**
   * Transform frontend farm data to API format
   */
  static transformToApiFormat(farmData: any): CreateFarmRequest {
    return {
      name: farmData.name,
      crop: farmData.crop,
      plantingDate: farmData.plantingDate,
      harvestDate: farmData.harvestDate,
      description: farmData.description,
      coordinates: farmData.coordinates,
      area: farmData.area,
    };
  }

  /**
   * Transform API response to frontend format
   */
  static transformFromApiFormat(apiData: any): Farm {
    let userId = apiData.userId;
    if (userId && typeof userId === 'object' && '_id' in userId) {
      userId = userId._id;
    }
    return {
      id: apiData._id || apiData.id,
      name: apiData.name,
      crop: apiData.crop,
      plantingDate: apiData.plantingDate,
      harvestDate: apiData.harvestDate,
      description: apiData.description,
      coordinates: Array.isArray(apiData.coordinates?.coordinates)
        ? apiData.coordinates.coordinates[0]
        : [],
      area: apiData.area,
      createdAt: apiData.createdAt,
      updatedAt: apiData.updatedAt,
      userId: userId,
    };
  }
}

export default FarmAPI;
