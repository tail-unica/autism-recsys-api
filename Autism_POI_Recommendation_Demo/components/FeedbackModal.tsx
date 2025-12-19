import { useState } from 'react';
import { X } from 'lucide-react';

interface FeedbackModalProps {
  placeName: string;
  liked: boolean;
  onClose: () => void;
  onSubmit: (feedback: { rating: number; comment: string }) => void;
}

export function FeedbackModal({ placeName, liked, onClose, onSubmit }: FeedbackModalProps) {
  const [rating, setRating] = useState<number | null>(null);
  const [comment, setComment] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (rating !== null) {
      onSubmit({ rating, comment });
      onClose();
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-40 flex items-center justify-center p-6 z-50">
      <div className="bg-[var(--color-bg-secondary)] rounded-2xl p-8 max-w-md w-full relative">
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)]"
        >
          <X size={24} />
        </button>

        <h2 className="mb-4">Feedback aggiuntivo</h2>
        <p className="mb-6">
          {placeName} - {liked ? 'Ti è piaciuto' : 'Non ti è piaciuto'}
        </p>

        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label className="block mb-3">Quanto consiglieresti questo posto?</label>
            <div className="flex gap-2 justify-between">
              {[1, 2, 3, 4, 5].map((value) => (
                <button
                  key={value}
                  type="button"
                  onClick={() => setRating(value)}
                  className={`flex-1 py-3 px-4 rounded-xl border-2 transition-all ${
                    rating === value
                      ? 'border-[var(--color-primary)] bg-[var(--color-bg-accent)] scale-105'
                      : 'border-[var(--color-border)] bg-white hover:border-[var(--color-primary)]'
                  }`}
                >
                  {value}
                </button>
              ))}
            </div>
            <div className="flex justify-between text-sm text-[var(--color-text-secondary)] mt-2 px-1">
              <span>Per niente</span>
              <span>Molto</span>
            </div>
          </div>

          <div>
            <label htmlFor="comment" className="block mb-2">
              Commento (opzionale)
            </label>
            <textarea
              id="comment"
              value={comment}
              onChange={(e) => setComment(e.target.value)}
              rows={4}
              placeholder="Cosa ti è piaciuto o non ti è piaciuto di questo posto?"
              className="w-full px-4 py-3 rounded-xl border-2 border-[var(--color-border)] bg-white focus:border-[var(--color-primary)] focus:outline-none transition-colors resize-none"
            />
          </div>

          <div className="flex gap-3">
            <button
              type="button"
              onClick={onClose}
              className="flex-1 py-3 px-6 border-2 border-[var(--color-border)] rounded-xl hover:bg-[var(--color-bg-accent)] transition-colors"
            >
              Salta
            </button>
            <button
              type="submit"
              disabled={rating === null}
              className={`flex-1 py-3 px-6 rounded-xl transition-all ${
                rating !== null
                  ? 'bg-[var(--color-primary)] hover:bg-[var(--color-primary-hover)] text-white'
                  : 'bg-[var(--color-border)] text-[var(--color-text-secondary)] cursor-not-allowed'
              }`}
            >
              Invia
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
