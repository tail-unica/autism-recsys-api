import { useState } from 'react';
import { sha256Hex } from '../lib/hash';

interface NicknameStepProps {
  initialNickname?: string;
  onComplete: (nickname: string, nicknameHash: string, isNewUser: boolean) => void;
}

export function NicknameStep({ initialNickname = '', onComplete }: NicknameStepProps) {
  const [nickname, setNickname] = useState(initialNickname);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    const normalizedNickname = nickname.trim();
    if (!normalizedNickname) {
      setError('Inserisci un nickname');
      return;
    }

    let nicknameHash: string;
    try {
      nicknameHash = await sha256Hex(normalizedNickname);
    } catch {
      setError('Impossibile calcolare l\'hash del nickname su questo browser');
      return;
    }

    // TODO: Check if user exists in Supabase
    // For now, simulate with localStorage
    const existingUsers = JSON.parse(localStorage.getItem('users') || '{}');

    // Backward-compat: migrate legacy storage keyed by plain nickname.
    if (existingUsers[normalizedNickname] && !existingUsers[nicknameHash]) {
      const legacy = { ...existingUsers[normalizedNickname] };
      delete legacy.nickname;
      existingUsers[nicknameHash] = {
        ...legacy,
        nicknameHash,
      };
      delete existingUsers[normalizedNickname];
      localStorage.setItem('users', JSON.stringify(existingUsers));
    }

    const isNewUser = !existingUsers[nicknameHash];

    if (isNewUser) {
      existingUsers[nicknameHash] = {
        nicknameHash,
        profile: null,
        createdAt: new Date().toISOString(),
      };
      localStorage.setItem('users', JSON.stringify(existingUsers));
    }

    onComplete(normalizedNickname, nicknameHash, isNewUser);
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

          <button
            type="submit"
            className="w-full bg-[var(--color-primary)] hover:bg-[var(--color-primary-hover)] text-white py-3 px-6 rounded-xl transition-colors"
          >
            Continua
          </button>
        </form>
      </div>
    </div>
  );
}
