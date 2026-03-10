// Chiamate API al backend con autenticazione

import { getAuthHeaders, clearToken } from './auth';
import { Place, Recommendation, SensoryFeatures, SensoryFeatureKey } from './types';
import { apiConfig } from '../resources/api_config';

const BACKEND_BASE = `${import.meta.env.BASE_URL}backend`;

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

// --- Normalizzazione dati dal backend ---
// Il server Express normalizza già i dati dell'API. Questi helper gestiscono
// solo il formato piatto restituito da MongoDB (sensory_features: {light: N, ...}).

const SENSORY_KEYS: readonly SensoryFeatureKey[] = ['light', 'space', 'crowd', 'noise', 'odor'];

const toCoordinateTuple = (value: unknown): [number, number] | undefined => {
  if (!Array.isArray(value) || value.length < 2) return undefined;
  const [a, b] = value;
  return typeof a === 'number' && typeof b === 'number' ? [a, b] : undefined;
};

const normalizeSensoryFeatures = (raw: unknown): SensoryFeatures => {
  if (!raw || typeof raw !== 'object') return {};
  const out: SensoryFeatures = {};
  for (const key of SENSORY_KEYS) {
    const val = (raw as Record<string, unknown>)[key];
    if (typeof val === 'number') out[key] = val;
  }
  return out;
};

const normalizeRecommendationFromBackend = (item: any, index: number): Recommendation => ({
  id: item.id ?? item.place ?? `rec-${index}`,
  name: item.name ?? item.place ?? `Luogo ${index + 1}`,
  address: item.address ?? '',
  category: item.category ?? undefined,
  description: item.description ?? undefined,
  sensory_features: normalizeSensoryFeatures(item.sensory_features),
  coordinates: toCoordinateTuple(item.coordinates),
  explanation: item.explanation ?? '',
  score: typeof item.score === 'number' ? item.score : undefined,
});

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
