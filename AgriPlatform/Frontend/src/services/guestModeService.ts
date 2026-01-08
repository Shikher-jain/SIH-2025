import { FarmAPI } from './farmApi';
import { GuestFarmStorage } from '../utils/guestFarmStorage';
import type { User } from './authApi';

/**
 * Service for managing guest mode functionality
 */
export class GuestModeService {
  /**
   * Create a guest user object
   */
  static createGuestUser(): User {
    return {
      id: 'guest',
      fullName: 'Guest User',
      email: 'guest@agriplatform.com',
      phone: '',
      address: '',
      role: 'user'
    };
  }

  /**
   * Check if current session is in guest mode
   */
  static isGuestMode(): boolean {
    const token = localStorage.getItem('auth_token');
    const user = localStorage.getItem('auth_user');
    
    // If no auth token but guest token exists, we're in guest mode
    const guestToken = localStorage.getItem('guest_mode');
    return !token && !user && guestToken === 'true';
  }

  /**
   * Enable guest mode
   */
  static enableGuestMode(): User {
    localStorage.setItem('guest_mode', 'true');
    const guestUser = this.createGuestUser();
    localStorage.setItem('guest_user', JSON.stringify(guestUser));
    return guestUser;
  }

  /**
   * Get guest user data
   */
  static getGuestUser(): User | null {
    try {
      const guestUserStr = localStorage.getItem('guest_user');
      return guestUserStr ? JSON.parse(guestUserStr) : null;
    } catch (error) {
      console.error('Error getting guest user:', error);
      return null;
    }
  }

  /**
   * Disable guest mode (cleanup)
   */
  static disableGuestMode(): void {
    localStorage.removeItem('guest_mode');
    localStorage.removeItem('guest_user');
  }

  /**
   * Migrate all guest farms to user account after signup/signin
   */
  static async migrateGuestFarmsToUser(): Promise<{ success: boolean; migratedCount: number; errors: string[] }> {
    const guestFarms = GuestFarmStorage.getAllForMigration();
    const errors: string[] = [];
    let migratedCount = 0;

    if (guestFarms.length === 0) {
      return { success: true, migratedCount: 0, errors: [] };
    }

    console.log(`🔄 Starting migration of ${guestFarms.length} guest farms...`);

    // Migrate each farm sequentially to avoid overwhelming the server
    for (const farmData of guestFarms) {
      try {
        const createRequest = FarmAPI.transformToApiFormat(farmData);
        const response = await FarmAPI.createFarm(createRequest);
        
        if (response.code === 1) {
          migratedCount++;
          console.log(`✅ Migrated farm: ${farmData.name}`);
        } else {
          const errorMsg = `Failed to migrate farm "${farmData.name}": ${response.message}`;
          console.error(`❌ ${errorMsg}`);
          errors.push(errorMsg);
        }
      } catch (error: any) {
        const errorMsg = `Failed to migrate farm "${farmData.name}": ${error.message || 'Unknown error'}`;
        console.error(`❌ ${errorMsg}`);
        errors.push(errorMsg);
      }

      // Add a small delay to prevent rate limiting
      await new Promise(resolve => setTimeout(resolve, 100));
    }

    const success = migratedCount > 0;
    
    if (success) {
      // Clear guest farms after successful migration
      GuestFarmStorage.clearAllFarms();
      // Disable guest mode
      this.disableGuestMode();
      console.log(`✅ Migration completed: ${migratedCount}/${guestFarms.length} farms migrated`);
    }

    return { success, migratedCount, errors };
  }

  /**
   * Check if user should be automatically logged in as guest
   */
  static shouldAutoEnableGuest(): boolean {
    const hasAuthToken = !!localStorage.getItem('auth_token');
    const hasAuthUser = !!localStorage.getItem('auth_user');
    const hasGuestMode = !!localStorage.getItem('guest_mode');
    
    // Enable guest mode if no authentication and not already in guest mode
    return !hasAuthToken && !hasAuthUser && !hasGuestMode;
  }

  /**
   * Get total guest farms count
   */
  static getGuestFarmsCount(): number {
    return GuestFarmStorage.getCount();
  }

  /**
   * Check if there are guest farms to migrate
   */
  static hasGuestFarmsToMigrate(): boolean {
    return GuestFarmStorage.hasGuestFarms();
  }
}

export default GuestModeService;