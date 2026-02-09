import mongoose from 'mongoose';
import crypto from 'crypto';

const userSchema = new mongoose.Schema({
  // Hash SHA-256 del nickname per identificazione sicura
  nicknameHash: {
    type: String,
    required: true,
    unique: true,
    index: true,
  },
  // Salt per derivazione del token (non salviamo il nickname in chiaro)
  tokenSalt: {
    type: String,
    required: true,
    default: () => crypto.randomBytes(32).toString('hex'),
  },
  // Profilo dell'utente (risposte alle domande)
  profile: {
    // Informazioni personali
    age: { type: Number, min: 0, max: 120 },
    asd: { type: Number, enum: [0, 1] }, // 0 = No, 1 = Sì
    // Avversioni sensoriali (1-5)
    bright_light: { type: Number, min: 1, max: 5 },
    dim_light: { type: Number, min: 1, max: 5 },
    crowd: { type: Number, min: 1, max: 5 },
    noise: { type: Number, min: 1, max: 5 },
    odor: { type: Number, min: 1, max: 5 },
    narrow_space: { type: Number, min: 1, max: 5 },
    wide_space: { type: Number, min: 1, max: 5 },
  },
  // Luoghi preferiti selezionati dall'utente
  favoritePlaces: [{
    id: String,
    name: String,
    address: String,
    category: String,
    image: String,
    description: String,
    coordinates: {
      type: [Number], // [longitude, latitude]
      index: '2dsphere',
    },
    sensory_features: {
      light: Number,
      space: Number,
      crowd: Number,
      noise: Number,
      odor: Number,
    },
  }],
  // Metadati
  createdAt: {
    type: Date,
    default: Date.now,
  },
  updatedAt: {
    type: Date,
    default: Date.now,
  },
  lastLoginAt: {
    type: Date,
    default: Date.now,
  },
});

// Aggiorna il timestamp quando il documento viene modificato
userSchema.pre('save', function(next) {
  this.updatedAt = new Date();
  next();
});

// Metodo per aggiornare il profilo
userSchema.methods.updateProfile = function(profileAnswers) {
  this.profile = { ...this.profile, ...profileAnswers };
  return this.save();
};

// Metodo per aggiornare i luoghi preferiti
userSchema.methods.updateFavoritePlaces = function(places) {
  this.favoritePlaces = places;
  return this.save();
};

export default mongoose.model('User', userSchema);
