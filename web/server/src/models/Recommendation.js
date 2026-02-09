import mongoose from 'mongoose';

const recommendationSchema = new mongoose.Schema({
  // Riferimento all'utente
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true,
    index: true,
  },
  // Hash del nickname (per query rapide)
  nicknameHash: {
    type: String,
    required: true,
    index: true,
  },
  // ID della sessione di raccomandazione
  sessionId: {
    type: String,
    required: true,
    index: true,
    default: () => new mongoose.Types.ObjectId().toString(),
  },
  // Richiesta inviata all'API
  request: {
    preferences: [String], // Nomi dei luoghi preferiti
    previousRecommendations: [String],
    recommendationCount: Number,
    diversityFactor: Number,
    restrictPreferences: Boolean,
    aversions: [{
      feature_name: String,
      rating: Number,
    }],
  },
  // Raccomandazioni ricevute dall'API
  recommendations: [{
    id: String,
    name: String,
    address: String,
    category: String,
    image: String,
    description: String,
    explanation: String,
    score: Number,
    coordinates: [Number],
    sensory_features: {
      light: Number,
      space: Number,
      crowd: Number,
      noise: Number,
      odor: Number,
    },
  }],
  // Fonte dei dati (api o mock)
  source: {
    type: String,
    enum: ['api', 'mock'],
    default: 'api',
  },
  // Timestamp
  createdAt: {
    type: Date,
    default: Date.now,
    index: true,
  },
});

// Index composto per query frequenti
recommendationSchema.index({ userId: 1, createdAt: -1 });
recommendationSchema.index({ nicknameHash: 1, createdAt: -1 });

export default mongoose.model('Recommendation', recommendationSchema);
