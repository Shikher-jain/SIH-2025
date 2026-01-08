// API related types

export interface ApiError {
  code: string;
  message: string;
  details?: Record<string, unknown>;
  statusCode?: number;
}

export interface ValidationError {
  field: string;
  message: string;
  code?: string;
}

export interface AppError {
  type: 'api' | 'validation' | 'network' | 'auth' | 'unknown';
  message: string;
  details?: Record<string, unknown>;
  statusCode?: number;
}

export interface RequestConfig {
  timeout?: number;
  retries?: number;
  retryDelay?: number;
  headers?: Record<string, string>;
}

export interface ApiEndpoints {
  auth: {
    login: string;
    register: string;
    logout: string;
    refresh: string;
    forgotPassword: string;
    resetPassword: string;
  };
  farms: {
    list: string;
    create: string;
    get: (id: string) => string;
    update: (id: string) => string;
    delete: (id: string) => string;
  };
  alerts: {
    list: string;
    markAsRead: (id: string) => string;
    dismiss: (id: string) => string;
  };
}
