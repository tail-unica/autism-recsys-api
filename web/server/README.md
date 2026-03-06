# Web Server Backend

Backend proxy per l'applicazione di raccomandazione POI, con autenticazione sicura e persistenza su MongoDB.

## Caratteristiche

- **Autenticazione sicura con nickname**: Login senza password usando hash SHA-256 del nickname e token JWT
- **Persistenza utenti**: Profilo utente, preferenze e luoghi preferiti salvati su MongoDB
- **Tracciamento raccomandazioni**: Ogni richiesta di raccomandazione viene salvata con il relativo sessionId
- **Feedback questionario**: Le risposte al questionario per ogni raccomandazione vengono salvate nel database
- **Validazione studio utente**: Sessioni marcate e salvate quando viene fornito un codice studio valido
- **Rate limiting**: Protezione contro abusi con rate limiting su tutte le API
- **Proxy API**: Proxy trasparente verso l'API di raccomandazione

## Architettura

```
web/
├── server/                  # Backend Node.js
│   ├── src/
│   │   ├── index.js        # Entry point Express
│   │   ├── middleware/
│   │   │   └── auth.js     # JWT authentication
│   │   ├── models/
│   │   │   ├── User.js          # Schema utente
│   │   │   ├── Recommendation.js # Schema raccomandazioni
│   │   │   └── Feedback.js      # Schema feedback
│   │   └── routes/
│   │       ├── auth.js          # Login/logout/verify
│   │       ├── user.js          # Profilo e preferiti
│   │       ├── recommendation.js # Richieste raccomandazioni
│   │       └── feedback.js      # Feedback questionario
│   └── package.json
├── lib/
│   ├── auth.ts             # Client auth (login, token management)
│   └── backend.ts          # Client API per backend
└── ...
```

## API Endpoints

### Autenticazione (`/auth`)

| Metodo | Endpoint | Descrizione |
|--------|----------|-------------|
| POST | `/auth/login` | Login/registrazione con nickname |
| POST | `/auth/verify` | Verifica validità token |
| POST | `/auth/logout` | Logout |

`POST /auth/login` accetta anche opzionalmente `studyCode` e `studySource` (`url` o `manual`).
Se `studyCode` è valido rispetto a `USER_STUDY_SECRET`, la sessione viene inserita in MongoDB nella collezione delle sessioni validate per lo studio.

### Utente (`/user`) - Richiede autenticazione

| Metodo | Endpoint | Descrizione |
|--------|----------|-------------|
| GET | `/user/profile` | Recupera profilo utente |
| PUT | `/user/profile` | Aggiorna profilo (risposte questionario) |
| GET | `/user/favorites` | Recupera luoghi preferiti |
| PUT | `/user/favorites` | Aggiorna luoghi preferiti |

### Raccomandazioni (`/recommendation`) - Richiede autenticazione

| Metodo | Endpoint | Descrizione |
|--------|----------|-------------|
| POST | `/recommendation/request` | Richiede raccomandazioni (salva su DB) |
| GET | `/recommendation/history` | Storico raccomandazioni utente |
| GET | `/recommendation/:sessionId` | Dettaglio sessione raccomandazione |

### Feedback (`/feedback`) - Richiede autenticazione

| Metodo | Endpoint | Descrizione |
|--------|----------|-------------|
| POST | `/feedback` | Invia feedback per una raccomandazione |
| GET | `/feedback/session/:sessionId` | Feedback per una sessione |
| GET | `/feedback/history` | Storico feedback utente |
| GET | `/feedback/stats` | Statistiche aggregate feedback |

## Sicurezza

### Autenticazione

1. L'utente inserisce un nickname
2. Il nickname viene normalizzato (lowercase, trim) e hashato con SHA-256
3. Il backend cerca/crea l'utente basandosi sull'hash
4. Viene generato un token JWT con:
   - `userId`: ID MongoDB dell'utente
   - `nicknameHash`: Hash del nickname
   - Scadenza configurabile (default: 7 giorni)

### Protezioni

- **Rate limiting**: 100 richieste/15min per auth, 60 richieste/min per altre API
- **Helmet**: Headers di sicurezza HTTP
- **CORS**: Configurabile per origine
- **JWT**: Token firmato con secret casuale
- **Hash nickname**: Il nickname non viene mai salvato in chiaro

## Schema Database

### User
```javascript
{
  nicknameHash: String,      // SHA-256 del nickname (unico)
  tokenSalt: String,         // Salt per sicurezza aggiuntiva
  profile: {
    age: Number,
    asd: Number,             // 0/1
    bright_light: Number,    // 1-5
    dim_light: Number,
    crowd: Number,
    noise: Number,
    odor: Number,
    narrow_space: Number,
    wide_space: Number
  },
  favoritePlaces: [{...}],
  createdAt: Date,
  updatedAt: Date,
  lastLoginAt: Date
}
```

### Recommendation
```javascript
{
  userId: ObjectId,
  nicknameHash: String,
  sessionId: String,         // ID univoco sessione
  request: {...},            // Payload inviato all'API
  recommendations: [{...}],  // Risultati ricevuti
  source: 'api' | 'mock',
  createdAt: Date
}
```

### Feedback
```javascript
{
  userId: ObjectId,
  nicknameHash: String,
  recommendationSessionId: String,
  placeId: String,
  placeName: String,
  liked: Boolean,
  surveyAnswers: {
    effectiveness: Number,
    decision_speed: Number,
    motivation: Number,
    satisfaction: Number,
    understanding: Number,
    trasparency: Number,
    confidence_boost: Number
  },
  comment: String,
  createdAt: Date
}
```

## Variabili d'Ambiente

```bash
PORT=3001                    # Porta del server
MONGODB_URI=mongodb://...    # URI connessione MongoDB
API_TARGET=http://api:8100   # URL API raccomandazioni
JWT_SECRET=...               # Secret per firma JWT (minimo 64 char)
USER_STUDY_SECRET=...        # Secret per validare codici studio
JWT_EXPIRES_IN=7d            # Durata token
CORS_ORIGIN=*                # Origini CORS consentite
```

## Sviluppo Locale

```bash
cd web/server
npm install
npm run dev
```

## Docker

Il server viene avviato automaticamente con `docker compose up`. Vedere il file `compose.yaml` per la configurazione.
