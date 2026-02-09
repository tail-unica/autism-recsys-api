import mongoose from 'mongoose';

const feedbackSchema = new mongoose.Schema({
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
  // Riferimento alla sessione di raccomandazione
  recommendationSessionId: {
    type: String,
    required: true,
    index: true,
  },
  // ID del luogo raccomandato
  placeId: {
    type: String,
    required: true,
  },
  // Nome del luogo (per riferimento)
  placeName: {
    type: String,
  },
  // L'utente ha gradito la spiegazione?
  liked: {
    type: Boolean,
    required: true,
  },
  // Risposte al questionario
  surveyAnswers: {
    effectiveness: { type: Number, min: 1, max: 5 },
    decision_speed: { type: Number, min: 1, max: 5 },
    motivation: { type: Number, min: 1, max: 5 },
    satisfaction: { type: Number, min: 1, max: 5 },
    understanding: { type: Number, min: 1, max: 5 },
    trasparency: { type: Number, min: 1, max: 5 },
    confidence_boost: { type: Number, min: 1, max: 5 },
  },
  // Commento libero dell'utente
  comment: {
    type: String,
    maxlength: 2000,
  },
  // Timestamp
  createdAt: {
    type: Date,
    default: Date.now,
    index: true,
  },
});

// Index composto per query frequenti
feedbackSchema.index({ userId: 1, createdAt: -1 });
feedbackSchema.index({ recommendationSessionId: 1, placeId: 1 });

export default mongoose.model('Feedback', feedbackSchema);
