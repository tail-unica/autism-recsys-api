import { apiConfig } from '../resources/api_config';
import categories from '../resources/categories.json';
import mockData from '../resources/mock_data.json';
import mockExplanations from '../resources/mock_explanations.json';
import { Place, Recommendation, SensoryFeatureKey, SensoryFeatures } from './types';

export type ApiSource = 'api' | 'mock';

const apiBase = apiConfig.baseUrl.replace(/\/$/, '');
const HEALTH_URL = `${apiBase}${apiConfig.endpoints.health}`;
const SEARCH_URL = `${apiBase}${apiConfig.endpoints.search}`;
const RECOMMEND_URL = `${apiBase}${apiConfig.endpoints.recommend}`;

const normalizeRating = (value: unknown): number => {
  if (typeof value === 'number') return value;
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : 3;
};

const toCoordinateTuple = (value: unknown): [number, number] | undefined => {
  if (!Array.isArray(value) || value.length < 2) return undefined;
  const [longitude, latitude] = value;
  return typeof longitude === 'number' && typeof latitude === 'number'
    ? [longitude, latitude]
    : undefined;
};

const normalizeFeatureName = (name: string): SensoryFeatureKey | null => {
  const key = name.toLowerCase();
  if (key === 'light' || key === 'space' || key === 'crowd' || key === 'noise' || key === 'odor') return key;
  return null;
};

const normalizeFeatureSet = (features: unknown): SensoryFeatures => {
  if (!features) return {};
  if (Array.isArray(features)) {
    return features.reduce<SensoryFeatures>((acc, curr) => {
      const featureKey = curr?.feature_name ?? curr?.featureName;
      const mapped = typeof featureKey === 'string' ? normalizeFeatureName(featureKey) : null;
      if (mapped) acc[mapped] = normalizeRating(curr?.rating);
      return acc;
    }, {});
  }
  if (typeof features === 'object') {
    return Object.entries(features as Record<string, unknown>).reduce<SensoryFeatures>((acc, [key, value]) => {
      const mapped = normalizeFeatureName(key);
      if (mapped) acc[mapped] = normalizeRating(value);
      return acc;
    }, {});
  }
  return {};
};

const pickExplanation = (index: number): string => {
  if (!Array.isArray(mockExplanations) || mockExplanations.length === 0) return 'Suggerimento basato sulle preferenze inserite.';
  return mockExplanations[index % mockExplanations.length].explanation;
};

const toPlaceFromMock = (entry: any, index: number): Place => {
  return {
    id: entry['name:token_seq'] || `mock-${index}`,
    name: entry['name:token_seq'] || `Mock Place ${index + 1}`,
    address: entry['address:token_seq'] || '',
    description: entry['description:token_seq'],
    image: entry['image_url:token_seq'] || apiConfig.fallback.placeholderImage,
    coordinates: toCoordinateTuple(entry['coordinates:float_seq']),
    sensory_features: normalizeFeatureSet(entry['sensory_features:token']),
    category: entry['category:token_seq'] || undefined,
  };
};

const toPlaceFromApi = (entry: any, index: number): Place => {
  const sensory = normalizeFeatureSet(entry?.sensory_features);
  return {
    id: entry?.place || `api-place-${index}`,
    name: entry?.place || `Luogo ${index + 1}`,
    address: entry?.address || '',
    category: entry?.category || undefined,
    image: entry?.image || apiConfig.fallback.placeholderImage,
    coordinates: toCoordinateTuple(entry?.coordinates?.geometry?.coordinates),
    sensory_features: sensory,
    description: entry?.description,
  };
};

const filterMockPlaces = (query: string, categoryIds?: string[]): Place[] => {
  const normalizedQuery = query.trim().toLowerCase();
  const categoryNames = categoryIds?.length
    ? categoryIds
    : undefined;
  return (mockData as any[])
    .map((entry, index) => toPlaceFromMock(entry, index))
    .filter((place) => {
      const matchesQuery = normalizedQuery
        ? place.name.toLowerCase().includes(normalizedQuery) || place.address?.toLowerCase().includes(normalizedQuery)
        : true;
      const matchesCategory = categoryNames
        ? categoryNames.some((cat) => place.category?.toLowerCase().includes(cat.toLowerCase()))
        : true;
      return matchesQuery && matchesCategory;
    });
};

const isApiAvailable = async (): Promise<boolean> => {
  try {
    const response = await fetch(HEALTH_URL);
    return response.ok;
  } catch (error) {
    console.warn('API non raggiungibile, uso mock.', error);
    return false;
  }
};

export const searchPlaces = async (params: {
  query: string;
  categories?: string[];
  limit?: number;
  distance?: number;
}): Promise<{ places: Place[]; source: ApiSource }> => {
  const query = params.query?.trim() || '';
  const categoryIds = params.categories?.filter(Boolean);
  const limit = params.limit || apiConfig.search.limit;
  const distance = params.distance || apiConfig.search.distance;

  const payload = {
    query,
    limit,
    distance,
    categories: categoryIds && categoryIds.length ? categoryIds : undefined,
  };

  if (await isApiAvailable()) {
    try {
      const response = await fetch(SEARCH_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (response.ok) {
        const data = await response.json();
        const places = Array.isArray(data?.results)
          ? data.results.map((entry: any, index: number) => toPlaceFromApi(entry, index))
          : [];
        if (places.length) return { places, source: 'api' };
      }
    } catch (error) {
      console.warn('Ricerca API fallita, uso mock.', error);
    }
  }

  const fallbackLimit = Math.max(1, limit - apiConfig.recommendations.count);
  return { places: filterMockPlaces(query, categoryIds).slice(0, fallbackLimit), source: 'mock' };
};

const buildAversions = (profileAnswers: Record<string, number>) => {
  const aversions: Record<string, number> = { ...apiConfig.aversionDefaults };
  Object.entries(apiConfig.profileQuestionToAversion).forEach(([questionId, aversionKey]) => {
    const answer = profileAnswers[questionId];
    if (typeof answer === 'number') {
      aversions[aversionKey] = answer;
    }
  });
  return Object.entries(aversions).map(([feature_name, rating]) => ({ feature_name, rating }));
};

const normalizeRecommendationFromApi = (item: any, index: number): Recommendation => {
  const metadata = item?.metadata || {};
  // Assicurati che sensory_features sia correttamente passato
  const placeEntry = {
    ...metadata,
    place: metadata.place || item.place,
    sensory_features: metadata.sensory_features || item.sensory_features
  };
  const place = toPlaceFromApi(placeEntry, index);
  return {
    ...place,
    explanation: item?.explanation || pickExplanation(index),
    score: typeof item?.score === 'number' ? item.score : undefined,
  };
};

const normalizeRecommendationFromMock = (entry: any, index: number): Recommendation => {
  const place = toPlaceFromMock(entry, index);
  return {
    ...place,
    explanation: pickExplanation(index),
  };
};

export const getRecommendations = async (params: {
  userId: string;
  profileAnswers: Record<string, number>;
  favoritePlaces: Place[];
  previousRecommendations?: string[];
}): Promise<{ items: Recommendation[]; source: ApiSource }> => {
  const payload = {
    user_id: params.userId || 'guest',
    preferences: params.favoritePlaces?.map((place) => place.name) || [],
    previous_recommendations: params.previousRecommendations || [],
    recommendation_count: apiConfig.recommendations.count,
    diversity_factor: apiConfig.recommendations.diversityFactor,
    restrict_preferences: apiConfig.recommendations.restrictPreferences,
    aversions: buildAversions(params.profileAnswers),
  };

  if (await isApiAvailable()) {
    try {
      const response = await fetch(RECOMMEND_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (response.ok) {
        const data = await response.json();
        const items = Array.isArray(data?.recommendations)
          ? data.recommendations.map((item: any, index: number) => normalizeRecommendationFromApi(item, index))
          : [];
        if (items.length) return { items, source: 'api' };
      }
    } catch (error) {
      console.warn('Chiamata /recommend fallita, uso mock.', error);
    }
  }

  const fallbackItems = [...(mockData as any[])]
    .reverse()
    .slice(0, apiConfig.recommendations.count)
    .map((entry, index) => normalizeRecommendationFromMock(entry, index));
  return { items: fallbackItems, source: 'mock' };
};

export const availableCategories = () => categories.map((c) => ({ id: c.id, name: c.name }));
