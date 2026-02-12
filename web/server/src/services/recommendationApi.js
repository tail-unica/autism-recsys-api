/**
 * Servizio per la comunicazione con l'API di raccomandazione.
 * Gestisce: chiamata HTTP, logging della risposta, normalizzazione dei dati.
 */

const API_TARGET = process.env.API_TARGET || 'http://api:8100';

const SENSORY_KEYS = new Set(['light', 'space', 'crowd', 'noise', 'odor']);

/**
 * Normalizza sensory_features da qualsiasi formato API a oggetto piatto { light: N, ... }
 *
 * Formati gestiti:
 *  1. Array:     [{ feature_name: "light", rating: 3 }, ...]
 *  2. Wrapper:   { features: [{ feature_name: "light", rating: 3 }, ...] }  (dummy)
 *  3. Oggetto:   { light: 3, space: 4, ... }
 */
export function normalizeSensoryFeatures(raw) {
  if (!raw) return {};

  // Formato wrapper del dummy: { features: [...] }
  if (!Array.isArray(raw) && typeof raw === 'object' && Array.isArray(raw.features)) {
    return normalizeSensoryFeatures(raw.features);
  }

  // Formato array: [{ feature_name, rating }, ...]
  if (Array.isArray(raw)) {
    const out = {};
    for (const item of raw) {
      const key = (item?.feature_name || item?.feature || '').toString().toLowerCase();
      const val = item?.rating ?? item?.value;
      if (SENSORY_KEYS.has(key) && val != null) out[key] = Number(val);
    }
    return out;
  }

  // Formato oggetto piatto: { light: 3, ... }
  if (typeof raw === 'object') {
    const out = {};
    for (const [k, v] of Object.entries(raw)) {
      const key = k.toLowerCase();
      if (SENSORY_KEYS.has(key) && v != null) out[key] = Number(v);
    }
    return out;
  }

  return {};
}

/**
 * Normalizza un singolo item di raccomandazione dall'API.
 * Risolve metadata e flatten dei campi.
 */
export function normalizeRecommendationItem(r, index = 0) {
  const meta = r?.metadata ?? {};
  const source = Object.keys(meta).length ? meta : r;

  return {
    id: source.place || r.place || `rec-${Date.now()}-${index}`,
    name: source.place || r.place || 'Luogo',
    address: source.address || r.address || '',
    category: source.category || r.category || '',
    image: source.image || r.image || '',
    description: source.description || r.description || '',
    explanation: r.explanation || '',
    score: r.score,
    coordinates: source.coordinates?.geometry?.coordinates || r.coordinates || [],
    sensory_features: normalizeSensoryFeatures(source.sensory_features || r.sensory_features),
  };
}

/**
 * Chiama l'API di raccomandazione e restituisce le raccomandazioni normalizzate.
 *
 * @param {Object} payload - Payload da inviare all'API (/recommend)
 * @returns {{ recommendations: Array, source: string }} - Raccomandazioni normalizzate e sorgente
 */
export async function fetchRecommendations(payload) {
  try {
    const url = `${API_TARGET}/recommend`;
    console.log(`[RecommendationAPI] POST ${url}`);
    console.log('[RecommendationAPI] Request payload:', JSON.stringify(payload, null, 2));

    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      console.warn(`[RecommendationAPI] Risposta non OK: ${response.status} ${response.statusText}`);
      const errorBody = await response.text().catch(() => '(unreadable)');
      console.warn('[RecommendationAPI] Error body:', errorBody);
      return { recommendations: [], source: 'mock' };
    }

    const data = await response.json();
    console.log('[RecommendationAPI] Raw response:', JSON.stringify(data, null, 2));

    const recommendations = Array.isArray(data?.recommendations)
      ? data.recommendations.map((item, index) => normalizeRecommendationItem(item, index))
      : [];

    console.log(`[RecommendationAPI] ${recommendations.length} raccomandazioni normalizzate`);

    return { recommendations, source: 'api' };

  } catch (apiError) {
    console.error('[RecommendationAPI] Errore di connessione:', apiError.message);
    return { recommendations: [], source: 'mock' };
  }
}
