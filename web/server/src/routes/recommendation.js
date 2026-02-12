import { Router } from 'express';
import Recommendation from '../models/Recommendation.js';
import User from '../models/User.js';
import { fetchRecommendations } from '../services/recommendationApi.js';

const router = Router();

/**
 * POST /recommendation/request
 * Richiede raccomandazioni all'API e le persiste nel database.
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

    // --- Logica API (delegata al servizio) ---
    const apiPayload = {
      user_id: req.user.nicknameHash,
      preferences: preferences || user.favoritePlaces?.map(p => p.name) || [],
      previous_recommendations: previousRecommendations || [],
      recommendation_count: recommendationCount,
      diversity_factor: diversityFactor,
      restrict_preferences: restrictPreferences,
      aversions,
    };

    const { recommendations, source } = await fetchRecommendations(apiPayload);

    // --- Logica MongoDB (persistenza) ---
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
      recommendations,
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
