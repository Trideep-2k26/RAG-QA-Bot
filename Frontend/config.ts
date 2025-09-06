export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export const API_ENDPOINTS = {
  upload: `${API_BASE_URL}/upload`,
  ask: `${API_BASE_URL}/ask`,
};
