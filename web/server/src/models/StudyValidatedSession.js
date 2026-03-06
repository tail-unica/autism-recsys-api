import mongoose from 'mongoose';

const studyValidatedSessionSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true,
    index: true,
  },
  nicknameHash: {
    type: String,
    required: true,
    index: true,
  },
  tokenId: {
    type: String,
    required: true,
    index: true,
  },
  recommendationSessionId: {
    type: String,
    default: '',
    index: true,
  },
  source: {
    type: String,
    enum: ['url', 'manual'],
    default: 'manual',
  },
  studyCodeHash: {
    type: String,
    required: true,
  },
  ipAddress: {
    type: String,
    default: '',
  },
  userAgent: {
    type: String,
    default: '',
  },
  loginAt: {
    type: Date,
    default: Date.now,
    index: true,
  },
  lastSeenAt: {
    type: Date,
    default: Date.now,
  },
  expiresAt: {
    type: Date,
    index: true,
  },
  revokedAt: {
    type: Date,
    default: null,
    index: true,
  },
});

studyValidatedSessionSchema.index(
  { tokenId: 1, recommendationSessionId: 1 },
  { unique: true }
);

export default mongoose.model('StudyValidatedSession', studyValidatedSessionSchema);
