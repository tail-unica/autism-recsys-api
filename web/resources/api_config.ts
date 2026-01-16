export const apiConfig = {
  baseUrl: '/api',
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
    count: 4,
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
      'https://upload.wikimedia.org/wikipedia/commons/thumb/d/db/20200325_Bellis_perennis_02.jpg/640px-20200325_Bellis_perennis_02.jpg',
  },
};
