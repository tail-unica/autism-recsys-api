import { Router } from 'express';
import crypto from 'crypto';
import User from '../models/User.js';
import StudyValidatedSession from '../models/StudyValidatedSession.js';
import { authenticateToken, generateToken, hashNickname, verifyToken as verifyJwtToken } from '../middleware/auth.js';

const router = Router();

const normalizeStudyCode = (code) => (typeof code === 'string' ? code.trim() : '');

const safeEqual = (left, right) => {
  const leftBuffer = Buffer.from(left);
  const rightBuffer = Buffer.from(right);
  if (leftBuffer.length !== rightBuffer.length) {
    return false;
  }

  return crypto.timingSafeEqual(leftBuffer, rightBuffer);
};

const buildPublicStudyCode = (secret) => {
  return crypto
    .createHmac('sha256', secret)
    .update('phase-user-study-link')
    .digest('base64url')
    .replace(/[-_]/g, '')
    .slice(0, 20)
    .toUpperCase();
};

const isValidStudyCode = (providedCode) => {
  const studySecret = normalizeStudyCode(process.env.USER_STUDY_SECRET);
  if (!studySecret || !providedCode) {
    return false;
  }

  const expectedPublicCode = buildPublicStudyCode(studySecret);
  return safeEqual(providedCode, studySecret) || safeEqual(providedCode, expectedPublicCode);
};

/**
 * POST /auth/login
 * Login o registrazione con nickname.
 * Sicurezza: il nickname viene hashato e non salvato in chiaro.
 * Un token JWT viene generato usando un salt unico per utente.
 */
router.post('/login', async (req, res) => {
  try {
    const { nickname, studyCode, studySource } = req.body;

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

    const normalizedStudyCode = normalizeStudyCode(studyCode);
    const hasProvidedStudyCode = normalizedStudyCode.length > 0;
    const hasValidStudyCode = isValidStudyCode(normalizedStudyCode);

    if (hasProvidedStudyCode && !hasValidStudyCode) {
      return res.status(400).json({ error: 'Token di validazione non valido' });
    }

    const tokenId = crypto.randomUUID();
    // Genera token JWT
    const token = generateToken(user._id.toString(), nicknameHash, tokenId);

    if (hasValidStudyCode) {
      const decodedToken = verifyJwtToken(token);
      await StudyValidatedSession.create({
        userId: user._id,
        nicknameHash,
        tokenId,
        source: studySource === 'url' ? 'url' : 'manual',
        studyCodeHash: hashNickname(normalizedStudyCode),
        ipAddress: req.ip || '',
        userAgent: req.get('user-agent') || '',
        expiresAt: decodedToken?.exp ? new Date(decodedToken.exp * 1000) : undefined,
      });
    }

    res.json({
      token,
      isNewUser,
      hasProfile: !!(user.profile && Object.keys(user.profile).length > 0),
      profile: user.profile || null,
      favoritePlaces: user.favoritePlaces || [],
      isStudyValidated: hasValidStudyCode,
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

    const decoded = verifyJwtToken(token);

    if (!decoded) {
      return res.status(401).json({ valid: false });
    }

    // Verifica che l'utente esista ancora
    const user = await User.findById(decoded.userId);
    if (!user) {
      return res.status(401).json({ valid: false });
    }

    if (decoded.tokenId) {
      await StudyValidatedSession.updateMany(
        { tokenId: decoded.tokenId, revokedAt: null },
        { $set: { lastSeenAt: new Date() } }
      );
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
router.post('/logout', authenticateToken, async (req, res) => {
  if (req.user?.tokenId) {
    await StudyValidatedSession.updateMany(
      { tokenId: req.user.tokenId, revokedAt: null },
      { $set: { revokedAt: new Date(), lastSeenAt: new Date() } }
    );
  }

  res.json({ success: true });
});

export default router;
