import { useEffect, useState } from 'react';
import { sha256Hex } from '../lib/hash';
import { login } from '../lib/auth';
import { Place } from '../lib/types';

interface NicknameStepProps {
  initialNickname?: string;
  initialStudyCode?: string;
  onComplete: (nickname: string, nicknameHash: string, isNewUser: boolean, profile?: Record<string, number>, favoritePlaces?: Place[]) => void;
}

export function NicknameStep({ initialNickname = '', initialStudyCode = '', onComplete }: NicknameStepProps) {
  const [nickname, setNickname] = useState(initialNickname);
  const [studyCode, setStudyCode] = useState(initialStudyCode);
  const [error, setError] = useState('');
  const [studyCodeError, setStudyCodeError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (!initialStudyCode) return;
    setStudyCode((currentValue) => currentValue || initialStudyCode);
  }, [initialStudyCode]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setStudyCodeError('');

    const normalizedNickname = nickname.trim();
    if (!normalizedNickname) {
      setError('Inserisci un nickname');
      return;
    }

    if (normalizedNickname.length < 2 || normalizedNickname.length > 50) {
      setError('Il nickname deve essere tra 2 e 50 caratteri');
      return;
    }

    const normalizedStudyCode = studyCode.trim();
    const studySource = normalizedStudyCode
      ? (normalizedStudyCode === initialStudyCode.trim() ? 'url' : 'manual')
      : undefined;

    setIsLoading(true);

    try {
      // Login tramite backend
      const response = await login(
        normalizedNickname,
        normalizedStudyCode || undefined,
        studySource
      );
      
      // Calcola l'hash anche lato client per uso interno
      let nicknameHash: string;
      try {
        nicknameHash = await sha256Hex(normalizedNickname.toLowerCase());
      } catch {
        nicknameHash = response.token.substring(0, 64); // Fallback
      }

      onComplete(
        normalizedNickname,
        nicknameHash,
        response.isNewUser,
        response.profile || undefined,
        response.favoritePlaces || undefined
      );
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Errore durante il login';
      if (normalizedStudyCode && /token di validazione non valido|codice studio non valido/i.test(message)) {
        setStudyCodeError('Token di validazione non valido');
      } else {
        setError(message);
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-6">
      <div className="bg-[var(--color-bg-secondary)] rounded-3xl shadow-sm p-8 md:p-12 max-w-md w-full">
        <h1 className="text-center mb-3">Benvenuto</h1>
        <p className="text-center mb-8">Inserisci il tuo nickname per iniziare</p>

        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label htmlFor="nickname" className="block mb-2">
              Nickname
            </label>
            <input
              id="nickname"
              type="text"
              value={nickname}
              onChange={(e) => setNickname(e.target.value)}
              className="w-full px-4 py-3 rounded-xl border-2 border-[var(--color-border)] bg-white focus:border-[var(--color-primary)] focus:outline-none transition-colors"
              placeholder="Il tuo nickname"
              autoFocus
            />
            {error && <p className="text-[var(--color-error)] mt-2">{error}</p>}
          </div>

          <div>
            <label htmlFor="studyCode" className="block mb-2">
              Codice studio (opzionale)
            </label>
            <input
              id="studyCode"
              type="text"
              value={studyCode}
              onChange={(e) => {
                setStudyCode(e.target.value);
                if (studyCodeError) {
                  setStudyCodeError('');
                }
              }}
              className="w-full px-4 py-3 rounded-xl border-2 border-[var(--color-border)] bg-white focus:border-[var(--color-primary)] focus:outline-none transition-colors"
              placeholder="Inserisci il codice, se disponibile"
            />
            {studyCodeError && <p className="text-[var(--color-error)] mt-2">{studyCodeError}</p>}
            {initialStudyCode && (
              <p className="text-sm text-[var(--color-text-secondary)] mt-2">
                Codice studio rilevato dal link.
              </p>
            )}
          </div>

          <button
            type="submit"
            disabled={isLoading}
            className={`w-full py-3 px-6 rounded-xl transition-colors ${
              isLoading
                ? 'bg-[var(--color-border)] text-[var(--color-text-secondary)] cursor-not-allowed'
                : 'bg-[var(--color-primary)] hover:bg-[var(--color-primary-hover)] text-white'
            }`}
          >
            {isLoading ? 'Accesso in corso...' : 'Continua'}
          </button>
        </form>
      </div>
    </div>
  );
}
