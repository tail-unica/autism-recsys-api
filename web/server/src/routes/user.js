import { Router } from 'express';
import User from '../models/User.js';

const router = Router();

/**
 * GET /user/profile
 * Recupera il profilo dell'utente corrente
 */
router.get('/profile', async (req, res) => {
  try {
    const user = await User.findById(req.user.userId);
    
    if (!user) {
      return res.status(404).json({ error: 'Utente non trovato' });
    }

    res.json({
      profile: user.profile || {},
      favoritePlaces: user.favoritePlaces || [],
      createdAt: user.createdAt,
      updatedAt: user.updatedAt,
    });

  } catch (error) {
    console.error('Get profile error:', error);
    res.status(500).json({ error: 'Errore interno del server' });
  }
});

/**
 * PUT /user/profile
 * Aggiorna il profilo dell'utente (risposte alle domande)
 */
router.put('/profile', async (req, res) => {
  try {
    const { answers } = req.body;

    if (!answers || typeof answers !== 'object') {
      return res.status(400).json({ error: 'Risposte al profilo richieste' });
    }

    const user = await User.findById(req.user.userId);
    
    if (!user) {
      return res.status(404).json({ error: 'Utente non trovato' });
    }

    // Aggiorna il profilo
    user.profile = {
      ...user.profile,
      ...answers,
    };
    await user.save();

    res.json({
      success: true,
      profile: user.profile,
    });

  } catch (error) {
    console.error('Update profile error:', error);
    res.status(500).json({ error: 'Errore interno del server' });
  }
});

/**
 * PUT /user/favorites
 * Aggiorna i luoghi preferiti dell'utente
 */
router.put('/favorites', async (req, res) => {
  try {
    const { places } = req.body;

    if (!Array.isArray(places)) {
      return res.status(400).json({ error: 'Lista dei luoghi richiesta' });
    }

    const user = await User.findById(req.user.userId);
    
    if (!user) {
      return res.status(404).json({ error: 'Utente non trovato' });
    }

    // Normalizza i luoghi
    const normalizedPlaces = places.map(place => ({
      id: place.id,
      name: place.name,
      address: place.address,
      category: place.category,
      image: place.image,
      description: place.description,
      coordinates: place.coordinates,
      sensory_features: place.sensory_features,
    }));

    user.favoritePlaces = normalizedPlaces;
    await user.save();

    res.json({
      success: true,
      favoritePlaces: user.favoritePlaces,
    });

  } catch (error) {
    console.error('Update favorites error:', error);
    res.status(500).json({ error: 'Errore interno del server' });
  }
});

/**
 * GET /user/favorites
 * Recupera i luoghi preferiti dell'utente
 */
router.get('/favorites', async (req, res) => {
  try {
    const user = await User.findById(req.user.userId);
    
    if (!user) {
      return res.status(404).json({ error: 'Utente non trovato' });
    }

    res.json({
      favoritePlaces: user.favoritePlaces || [],
    });

  } catch (error) {
    console.error('Get favorites error:', error);
    res.status(500).json({ error: 'Errore interno del server' });
  }
});

export default router;
