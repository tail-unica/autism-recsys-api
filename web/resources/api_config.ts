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

/**
 * Costruisce la lista di avversioni sensoriali a partire dalle risposte al profilo.
 * Usata sia dal frontend (RecommendationsStep) sia potenzialmente dal backend.
 */
export const buildAversions = (profileAnswers: Record<string, number>) => {
  const aversions: Record<string, number> = { ...apiConfig.aversionDefaults };
  // Mappa le domande del profilo alle avversioni corrispondenti
  Object.entries(apiConfig.profileQuestionToAversion).forEach(([questionId, aversionKey]) => {
    const answer = profileAnswers[questionId];
    if (typeof answer === 'number') {
      aversions[aversionKey] = answer;
    }
  });
  // Sovrascrive con risposte dirette alle avversioni (se presenti)
  const directKeys = ['bright_light', 'dim_light', 'crowd', 'noise', 'odor', 'narrow_space', 'wide_space'];
  directKeys.forEach(key => {
    if (typeof profileAnswers[key] === 'number') {
      aversions[key] = profileAnswers[key];
    }
  });
  return Object.entries(aversions).map(([feature_name, rating]) => ({ feature_name, rating }));
};
