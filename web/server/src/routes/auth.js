import { Router } from 'express';
import crypto from 'crypto';
import User from '../models/User.js';
import { generateToken, hashNickname } from '../middleware/auth.js';

const router = Router();

/**
 * POST /auth/login
 * Login o registrazione con nickname.
 * Sicurezza: il nickname viene hashato e non salvato in chiaro.
 * Un token JWT viene generato usando un salt unico per utente.
 */
router.post('/login', async (req, res) => {
  try {
    const { nickname } = req.body;

    if (!nickname || typeof nickname !== 'string') {
      return res.status(400).json({ error: 'Nickname richiesto' });
    }

    const normalizedNickname = nickname.trim().toLowerCase();
    if (normalizedNickname.length < 2 || normalizedNickname.length > 50) {
      return res.status(400).json({ error: 'Il nickname deve essere tra 2 e 50 caratteri' });
    }

    // Calcola l'hash del nickname
    const nicknameHash = hashNickname(normalizedNickname);

    // Cerca l'utente o creane uno nuovo
    let user = await User.findOne({ nicknameHash });
    let isNewUser = false;

    if (!user) {
      // Nuovo utente
      user = new User({
        nicknameHash,
        tokenSalt: crypto.randomBytes(32).toString('hex'),
      });
      await user.save();
      isNewUser = true;
    } else {
      // Aggiorna last login
      user.lastLoginAt = new Date();
      await user.save();
    }

    // Genera token JWT
    const token = generateToken(user._id.toString(), nicknameHash);

    res.json({
      token,
      isNewUser,
      hasProfile: !!(user.profile && Object.keys(user.profile).length > 0),
      profile: user.profile || null,
      favoritePlaces: user.favoritePlaces || [],
    });

  } catch (error) {
    console.error('Login error:', error);
    res.status(500).json({ error: 'Errore interno del server' });
  }
});

/**
 * POST /auth/verify
 * Verifica se il token è ancora valido
 */
router.post('/verify', async (req, res) => {
  try {
    const authHeader = req.headers['authorization'];
    const token = authHeader && authHeader.split(' ')[1];

    if (!token) {
      return res.status(401).json({ valid: false });
    }

    // Import verifyToken here to avoid circular dependency
    const { verifyToken } = await import('../middleware/auth.js');
    const decoded = verifyToken(token);

    if (!decoded) {
      return res.status(401).json({ valid: false });
    }

    // Verifica che l'utente esista ancora
    const user = await User.findById(decoded.userId);
    if (!user) {
      return res.status(401).json({ valid: false });
    }

    res.json({
      valid: true,
      hasProfile: !!(user.profile && Object.keys(user.profile).length > 0),
    });

  } catch (error) {
    console.error('Verify error:', error);
    res.status(500).json({ error: 'Errore interno del server' });
  }
});

/**
 * POST /auth/logout
 * Logout (lato client, invalida il token)
 */
router.post('/logout', (req, res) => {
  // Il logout è gestito lato client rimuovendo il token
  // Qui potremmo aggiungere una blacklist di token se necessario
  res.json({ success: true });
});

export default router;
