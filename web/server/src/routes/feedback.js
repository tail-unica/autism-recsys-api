import { Router } from 'express';
import Feedback from '../models/Feedback.js';
import Recommendation from '../models/Recommendation.js';

const router = Router();

/**
 * POST /feedback
 * Salva il feedback dell'utente per una raccomandazione
 */
router.post('/', async (req, res) => {
  try {
    const {
      sessionId,
      placeId,
      placeName,
      liked,
      answers,
      comment,
    } = req.body;

    // Validazione
    if (!sessionId || !placeId) {
      return res.status(400).json({ error: 'sessionId e placeId sono richiesti' });
    }

    if (typeof liked !== 'boolean') {
      return res.status(400).json({ error: 'liked deve essere un booleano' });
    }

    // Verifica che la sessione di raccomandazione esista
    const recommendation = await Recommendation.findOne({
      sessionId,
      userId: req.user.userId,
    });

    if (!recommendation) {
      return res.status(404).json({ error: 'Sessione di raccomandazione non trovata' });
    }

    // Verifica che il placeId sia tra le raccomandazioni della sessione
    const placeExists = recommendation.recommendations.some(r => r.id === placeId);
    if (!placeExists) {
      return res.status(400).json({ error: 'Il luogo non fa parte di questa sessione di raccomandazione' });
    }

    // Verifica se esiste già un feedback per questo luogo nella sessione
    const existingFeedback = await Feedback.findOne({
      recommendationSessionId: sessionId,
      placeId,
      userId: req.user.userId,
    });

    if (existingFeedback) {
      // Aggiorna il feedback esistente
      existingFeedback.liked = liked;
      existingFeedback.surveyAnswers = answers || {};
      existingFeedback.comment = comment || '';
      await existingFeedback.save();

      return res.json({
        success: true,
        feedbackId: existingFeedback._id,
        updated: true,
      });
    }

    // Crea nuovo feedback
    const feedback = new Feedback({
      userId: req.user.userId,
      nicknameHash: req.user.nicknameHash,
      recommendationSessionId: sessionId,
      placeId,
      placeName: placeName || '',
      liked,
      surveyAnswers: answers || {},
      comment: comment || '',
    });

    await feedback.save();

    res.status(201).json({
      success: true,
      feedbackId: feedback._id,
      updated: false,
    });

  } catch (error) {
    console.error('Save feedback error:', error);
    res.status(500).json({ error: 'Errore interno del server' });
  }
});

/**
 * GET /feedback/session/:sessionId
 * Recupera tutti i feedback per una sessione di raccomandazione
 */
router.get('/session/:sessionId', async (req, res) => {
  try {
    const feedbacks = await Feedback.find({
      recommendationSessionId: req.params.sessionId,
      userId: req.user.userId,
    }).select('-__v');

    res.json({ feedbacks });

  } catch (error) {
    console.error('Get session feedbacks error:', error);
    res.status(500).json({ error: 'Errore interno del server' });
  }
});

/**
 * GET /feedback/history
 * Recupera lo storico dei feedback dell'utente
 */
router.get('/history', async (req, res) => {
  try {
    const limit = Math.min(parseInt(req.query.limit) || 20, 100);
    const skip = parseInt(req.query.skip) || 0;

    const feedbacks = await Feedback.find({ userId: req.user.userId })
      .sort({ createdAt: -1 })
      .skip(skip)
      .limit(limit)
      .select('-__v');

    const total = await Feedback.countDocuments({ userId: req.user.userId });

    res.json({
      feedbacks,
      total,
      limit,
      skip,
    });

  } catch (error) {
    console.error('Get feedback history error:', error);
    res.status(500).json({ error: 'Errore interno del server' });
  }
});

/**
 * GET /feedback/stats
 * Statistiche aggregate sui feedback dell'utente
 */
router.get('/stats', async (req, res) => {
  try {
    const stats = await Feedback.aggregate([
      { $match: { nicknameHash: req.user.nicknameHash } },
      {
        $group: {
          _id: null,
          totalFeedbacks: { $sum: 1 },
          likedCount: { $sum: { $cond: ['$liked', 1, 0] } },
          dislikedCount: { $sum: { $cond: ['$liked', 0, 1] } },
          avgEffectiveness: { $avg: '$surveyAnswers.effectiveness' },
          avgDecisionSpeed: { $avg: '$surveyAnswers.decision_speed' },
          avgMotivation: { $avg: '$surveyAnswers.motivation' },
          avgSatisfaction: { $avg: '$surveyAnswers.satisfaction' },
          avgUnderstanding: { $avg: '$surveyAnswers.understanding' },
          avgTrasparency: { $avg: '$surveyAnswers.trasparency' },
          avgConfidenceBoost: { $avg: '$surveyAnswers.confidence_boost' },
        },
      },
    ]);

    res.json(stats[0] || {
      totalFeedbacks: 0,
      likedCount: 0,
      dislikedCount: 0,
    });

  } catch (error) {
    console.error('Get feedback stats error:', error);
    res.status(500).json({ error: 'Errore interno del server' });
  }
});

export default router;
