import { useState } from 'react';

interface NicknameStepProps {
  onComplete: (nickname: string, isNewUser: boolean) => void;
}

export function NicknameStep({ onComplete }: NicknameStepProps) {
  const [nickname, setNickname] = useState('');
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (!nickname.trim()) {
      setError('Inserisci un nickname');
      return;
    }

    // TODO: Check if user exists in Supabase
    // For now, simulate with localStorage
    const existingUsers = JSON.parse(localStorage.getItem('users') || '{}');
    const isNewUser = !existingUsers[nickname];

    if (isNewUser) {
      existingUsers[nickname] = {
        nickname,
        profile: null,
        createdAt: new Date().toISOString(),
      };
      localStorage.setItem('users', JSON.stringify(existingUsers));
    }

    onComplete(nickname, isNewUser);
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
