import { Router } from 'express';
import Recommendation from '../models/Recommendation.js';
import User from '../models/User.js';

const router = Router();

const API_TARGET = process.env.API_TARGET || 'http://api:8100';

/**
 * Normalizza sensory_features da array [{feature_name, rating}] a oggetto {light: N, ...}
 */
function normalizeSensoryFeatures(raw) {
  if (!raw) return {};
  if (Array.isArray(raw)) {
    const out = {};
    for (const item of raw) {
      const key = (item?.feature_name || item?.feature || '').toString().toLowerCase();
      const val = item?.rating ?? item?.value;
      if (key && val != null) out[key] = Number(val);
    }
    return out;
  }
  if (typeof raw === 'object') return raw;
  return {};
}

/**
 * POST /recommendation/request
 * Richiede raccomandazioni e le salva nel database
 */
router.post('/request', async (req, res) => {
  try {
    const { 
      preferences,
      previousRecommendations,
      recommendationCount = 4,
      diversityFactor = 0.5,
      restrictPreferences = false,
      aversions = [],
    } = req.body;

    const user = await User.findById(req.user.userId);
    if (!user) {
      return res.status(404).json({ error: 'Utente non trovato' });
    }

    // Prepara la richiesta per l'API
    const apiPayload = {
      user_id: req.user.nicknameHash,
      preferences: preferences || user.favoritePlaces?.map(p => p.name) || [],
      previous_recommendations: previousRecommendations || [],
      recommendation_count: recommendationCount,
      diversity_factor: diversityFactor,
      restrict_preferences: restrictPreferences,
      aversions,
    };

    let recommendations = [];
    let source = 'api';

    try {
      // Chiama l'API di raccomandazione
      const response = await fetch(`${API_TARGET}/recommend`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(apiPayload),
      });

      if (response.ok) {
        const data = await response.json();
        recommendations = data.recommendations || [];
      } else {
        console.warn('API recommendation failed, status:', response.status);
        source = 'mock';
      }
    } catch (apiError) {
      console.warn('API recommendation error:', apiError.message);
      source = 'mock';
    }

    // Crea il record della raccomandazione nel database
    const recommendationRecord = new Recommendation({
      userId: user._id,
      nicknameHash: req.user.nicknameHash,
      request: {
        preferences: apiPayload.preferences,
        previousRecommendations: apiPayload.previous_recommendations,
        recommendationCount: apiPayload.recommendation_count,
        diversityFactor: apiPayload.diversity_factor,
        restrictPreferences: apiPayload.restrict_preferences,
        aversions: apiPayload.aversions,
      },
      recommendations: recommendations.map(r => ({
        id: r.metadata?.place || r.place || `rec-${Date.now()}`,
        name: r.metadata?.place || r.place || 'Luogo',
        address: r.metadata?.address || r.address || '',
        category: r.metadata?.category || r.category || '',
        image: r.metadata?.image || r.image || '',
        description: r.metadata?.description || r.description || '',
        explanation: r.explanation || '',
        score: r.score,
        coordinates: r.metadata?.coordinates?.geometry?.coordinates || r.coordinates || [],
        sensory_features: normalizeSensoryFeatures(r.metadata?.sensory_features || r.sensory_features),
      })),
      source,
    });

    await recommendationRecord.save();

    res.json({
      sessionId: recommendationRecord.sessionId,
      recommendations: recommendationRecord.recommendations,
      source,
    });

  } catch (error) {
    console.error('Recommendation request error:', error);
    res.status(500).json({ error: 'Errore interno del server' });
  }
});

/**
 * GET /recommendation/history
 * Recupera lo storico delle raccomandazioni dell'utente
 */
router.get('/history', async (req, res) => {
  try {
    const limit = Math.min(parseInt(req.query.limit) || 10, 100);
    const skip = parseInt(req.query.skip) || 0;

    const recommendations = await Recommendation.find({ userId: req.user.userId })
      .sort({ createdAt: -1 })
      .skip(skip)
      .limit(limit)
      .select('-__v');

    const total = await Recommendation.countDocuments({ userId: req.user.userId });

    res.json({
      recommendations,
      total,
      limit,
      skip,
    });

  } catch (error) {
    console.error('Get recommendation history error:', error);
    res.status(500).json({ error: 'Errore interno del server' });
  }
});

/**
 * GET /recommendation/:sessionId
 * Recupera una specifica sessione di raccomandazione
 */
router.get('/:sessionId', async (req, res) => {
  try {
    const recommendation = await Recommendation.findOne({
      sessionId: req.params.sessionId,
      userId: req.user.userId,
    });

    if (!recommendation) {
      return res.status(404).json({ error: 'Sessione non trovata' });
    }

    res.json(recommendation);

  } catch (error) {
    console.error('Get recommendation error:', error);
    res.status(500).json({ error: 'Errore interno del server' });
  }
});

export default router;
