export const apiConfig = {
  baseUrl: (typeof import.meta !== 'undefined' && (import.meta as any).env?.VITE_API_BASE_URL) || 'http://localhost:8000',
  endpoints: {
    health: '/health',
    search: '/search',
    recommend: '/recommend',
  },
  search: {
    limit: 12,
    distance: 1500,
  },
  recommendations: {
    count: 6,
    diversityFactor: 0.5,
    restrictPreferences: false,
  },
  aversionDefaults: {
    bright_light: 3,
    dim_light: 3,
    wide_space: 3,
    narrow_space: 3,
    crowd: 3,
    noise: 3,
    odor: 3,
  },
  profileQuestionToAversion: {
    noise_sensitivity: 'noise',
    crowd_comfort: 'crowd',
    light_sensitivity: 'bright_light',
  },
  fallback: {
    placeholderImage:
      'https://images.unsplash.com/photo-1505764706515-aa95265c5abc?auto=format&fit=crop&w=1200&q=80',
  },
};
