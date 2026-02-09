// Chiamate API al backend con autenticazione

import { getAuthHeaders, clearToken } from './auth';
import { Place, Recommendation, SensoryFeatures } from './types';
import { apiConfig } from '../resources/api_config';

const BACKEND_BASE = '/backend';

// Gestione errori di autenticazione
const handleAuthError = (response: Response) => {
  if (response.status === 401 || response.status === 403) {
    clearToken();
    window.location.reload();
  }
};

// Fetch con gestione errori
const authenticatedFetch = async (url: string, options: RequestInit = {}): Promise<Response> => {
  const response = await fetch(url, {
    ...options,
    headers: {
      ...getAuthHeaders(),
      ...options.headers,
    },
  });

  if (!response.ok) {
    handleAuthError(response);
  }

  return response;
};

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

const normalizeFeatureName = (name: string): 'light' | 'space' | 'crowd' | 'noise' | 'odor' | null => {
  const key = name.toLowerCase();
  if (key === 'light' || key === 'space' || key === 'crowd' || key === 'noise' || key === 'odor') return key;
  return null;
};

const normalizeFeatureSet = (features: unknown): SensoryFeatures => {
  if (!features) return {};
  if (Array.isArray(features)) {
    return features.reduce<SensoryFeatures>((acc, curr) => {
      const featureKey = curr?.feature_name ?? curr?.featureName ?? curr?.feature;
      const mapped = typeof featureKey === 'string' ? normalizeFeatureName(featureKey) : null;
      if (mapped) acc[mapped] = normalizeRating(curr?.rating ?? curr?.value);
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

type RawSensoryItem = { feature_name?: string; feature?: string; rating?: number; value?: number } | Record<string, number>;

const SENSORY_KEYS = ['odor', 'crowd', 'space', 'light', 'noise'] as const;

function normalizeSensoryFeatures(input: RawSensoryItem | RawSensoryItem[] | undefined) {
  const out: Record<string, number> = {};
  if (!input) return out;

  if (Array.isArray(input)) {
    for (const it of input) {
      const key = (it && (it.feature_name || (it as any).feature) || '').toString().toLowerCase();
      if (!key) continue;
      const val = (it as any).rating ?? (it as any).value;
      const num = typeof val === 'number' ? val : Number(val);
      if (!Number.isNaN(num) && SENSORY_KEYS.includes(key as any)) out[key] = num;
    }
    return out;
  }

  if (typeof input === 'object') {
    for (const k of Object.keys(input)) {
      const lk = k.toLowerCase();
      const v = (input as any)[k];
      const num = typeof v === 'number' ? v : Number(v);
      if (!Number.isNaN(num) && SENSORY_KEYS.includes(lk as any)) out[lk] = num;
    }
    return out;
  }

  return out;
}

function normalizeRecommendation(raw: any, index = 0) {
  const meta = raw?.metadata ?? {};
  const placeSource = Object.keys(meta).length ? meta : raw;
  const id = placeSource.placeId ?? placeSource.id ?? placeSource.place ?? raw.place ?? `api-place-${index}`;
  const name = placeSource.place ?? placeSource.name ?? raw.place ?? `Luogo ${index + 1}`;
  const address = placeSource.address ?? raw.address ?? '';
  const category = placeSource.category ?? raw.category;
  const image = placeSource.image ?? raw.image;
  const description = placeSource.description ?? raw.description;
  const sensory_features = normalizeSensoryFeatures(placeSource.sensory_features ?? raw.sensory_features);
  const coordinates = (placeSource.coordinates?.geometry?.coordinates ?? placeSource.coordinates ?? raw.coordinates) || undefined;

  return {
    id,
    name,
    address,
    category,
    image,
    description,
    sensory_features,
    coordinates,
    explanation: raw.explanation ?? '',
    score: typeof raw.score === 'number' ? raw.score : undefined,
  };
}

const normalizeRecommendationFromBackend = (item: any, index: number): Recommendation => {
  const metadata = item?.metadata || {};
  const coordinates =
    toCoordinateTuple(item?.coordinates) ||
    toCoordinateTuple(metadata?.coordinates?.geometry?.coordinates) ||
    toCoordinateTuple(item?.coordinates?.geometry?.coordinates);
  return normalizeRecommendation(item, index);
};

// ===== USER API =====

export interface UserProfile {
  profile: Record<string, number>;
  favoritePlaces: Place[];
  createdAt: string;
  updatedAt: string;
}

export const getUserProfile = async (): Promise<UserProfile> => {
  const response = await authenticatedFetch(`${BACKEND_BASE}/user/profile`);
  
  if (!response.ok) {
    throw new Error('Errore nel recupero del profilo');
  }

  return response.json();
};

export const updateUserProfile = async (answers: Record<string, number>): Promise<{ success: boolean; profile: Record<string, number> }> => {
  const response = await authenticatedFetch(`${BACKEND_BASE}/user/profile`, {
    method: 'PUT',
    body: JSON.stringify({ answers }),
  });

  if (!response.ok) {
    throw new Error('Errore nell\'aggiornamento del profilo');
  }

  return response.json();
};

export const updateFavoritePlaces = async (places: Place[]): Promise<{ success: boolean; favoritePlaces: Place[] }> => {
  const response = await authenticatedFetch(`${BACKEND_BASE}/user/favorites`, {
    method: 'PUT',
    body: JSON.stringify({ places }),
  });

  if (!response.ok) {
    throw new Error('Errore nell\'aggiornamento dei luoghi preferiti');
  }

  return response.json();
};

export const getFavoritePlaces = async (): Promise<{ favoritePlaces: Place[] }> => {
  const response = await authenticatedFetch(`${BACKEND_BASE}/user/favorites`);

  if (!response.ok) {
    throw new Error('Errore nel recupero dei luoghi preferiti');
  }

  return response.json();
};

// ===== RECOMMENDATION API =====

export interface RecommendationRequest {
  preferences?: string[];
  previousRecommendations?: string[];
  recommendationCount?: number;
  diversityFactor?: number;
  restrictPreferences?: boolean;
  aversions?: Array<{ feature_name: string; rating: number }>;
}

export interface RecommendationResponse {
  sessionId: string;
  recommendations: Recommendation[];
  source: 'api' | 'mock';
}

export const requestRecommendations = async (params: RecommendationRequest): Promise<RecommendationResponse> => {
  const response = await authenticatedFetch(`${BACKEND_BASE}/recommendation/request`, {
    method: 'POST',
    body: JSON.stringify({
      preferences: params.preferences,
      previousRecommendations: params.previousRecommendations,
      recommendationCount: params.recommendationCount || apiConfig.recommendations.count,
      diversityFactor: params.diversityFactor || apiConfig.recommendations.diversityFactor,
      restrictPreferences: params.restrictPreferences || apiConfig.recommendations.restrictPreferences,
      aversions: params.aversions,
    }),
  });

  if (!response.ok) {
    throw new Error('Errore nella richiesta delle raccomandazioni');
  }

  const data = await response.json();
  const recommendations = Array.isArray(data?.recommendations)
    ? data.recommendations.map((item: any, index: number) => normalizeRecommendationFromBackend(item, index))
    : [];
  return {
    sessionId: data?.sessionId || '',
    recommendations,
    source: data?.source || 'api',
  };
};

export const getRecommendationHistory = async (limit = 10, skip = 0): Promise<{
  recommendations: any[];
  total: number;
}> => {
  const response = await authenticatedFetch(
    `${BACKEND_BASE}/recommendation/history?limit=${limit}&skip=${skip}`
  );

  if (!response.ok) {
    throw new Error('Errore nel recupero dello storico');
  }

  return response.json();
};

// ===== FEEDBACK API =====

export interface FeedbackData {
  sessionId: string;
  placeId: string;
  placeName?: string;
  liked: boolean;
  answers: Record<string, number>;
  comment: string;
}

export const submitFeedback = async (feedback: FeedbackData): Promise<{ success: boolean; feedbackId: string }> => {
  const response = await authenticatedFetch(`${BACKEND_BASE}/feedback`, {
    method: 'POST',
    body: JSON.stringify(feedback),
  });

  if (!response.ok) {
    throw new Error('Errore nell\'invio del feedback');
  }

  return response.json();
};

export const getSessionFeedbacks = async (sessionId: string): Promise<{ feedbacks: any[] }> => {
  const response = await authenticatedFetch(`${BACKEND_BASE}/feedback/session/${sessionId}`);

  if (!response.ok) {
    throw new Error('Errore nel recupero dei feedback');
  }

  return response.json();
};

export const getFeedbackHistory = async (limit = 20, skip = 0): Promise<{
  feedbacks: any[];
  total: number;
}> => {
  const response = await authenticatedFetch(
    `${BACKEND_BASE}/feedback/history?limit=${limit}&skip=${skip}`
  );

  if (!response.ok) {
    throw new Error('Errore nel recupero dello storico feedback');
  }

  return response.json();
};

export const getFeedbackStats = async (): Promise<{
  totalFeedbacks: number;
  likedCount: number;
  dislikedCount: number;
  [key: string]: number;
}> => {
  const response = await authenticatedFetch(`${BACKEND_BASE}/feedback/stats`);

  if (!response.ok) {
    throw new Error('Errore nel recupero delle statistiche');
  }

  return response.json();
};
