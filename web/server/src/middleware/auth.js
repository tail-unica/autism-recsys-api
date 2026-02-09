import jwt from 'jsonwebtoken';
import crypto from 'crypto';

const JWT_SECRET = process.env.JWT_SECRET || crypto.randomBytes(64).toString('hex');
const JWT_EXPIRES_IN = process.env.JWT_EXPIRES_IN || '7d';

// Genera un token JWT per l'utente
export const generateToken = (userId, nicknameHash) => {
  return jwt.sign(
    { 
      userId, 
      nicknameHash,
      iat: Math.floor(Date.now() / 1000),
    },
    JWT_SECRET,
    { expiresIn: JWT_EXPIRES_IN }
  );
};

// Verifica un token JWT
export const verifyToken = (token) => {
  try {
    return jwt.verify(token, JWT_SECRET);
  } catch (error) {
    return null;
  }
};

// Middleware per autenticare le richieste
export const authenticateToken = (req, res, next) => {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1]; // Bearer TOKEN

  if (!token) {
    return res.status(401).json({ error: 'Token di autenticazione richiesto' });
  }

  const decoded = verifyToken(token);
  if (!decoded) {
    return res.status(403).json({ error: 'Token non valido o scaduto' });
  }

  req.user = decoded;
  next();
};

// Calcola l'hash SHA-256 del nickname
export const hashNickname = (nickname) => {
  return crypto.createHash('sha256').update(nickname).digest('hex');
};
