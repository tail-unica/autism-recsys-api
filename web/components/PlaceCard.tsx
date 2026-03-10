import { CheckCircle2 } from 'lucide-react';
import { Place } from '../lib/types';
import { renderMarkdownBold } from '../lib/markdown';

interface PlaceCardProps {
  place: Place;
  explanation?: string;
  isCompleted?: boolean;
  onClick?: () => void;
}

export function PlaceCard({ place, explanation, isCompleted, onClick }: PlaceCardProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`w-full text-left transition-colors rounded-2xl p-4 md:p-5 shadow-sm ${
        isCompleted
          ? 'bg-[var(--color-bg-accent)] border-2 border-[var(--color-primary)]'
          : 'bg-[var(--color-bg-secondary)] hover:bg-[var(--color-bg-accent)]'
      }`}
    >
      <div className="flex-1 min-w-0">
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0">
            <h3 className="mb-1 truncate flex items-center gap-2">
              {place.name}
              {isCompleted && (
                <span className="inline-flex items-center gap-1 text-xs font-medium text-[var(--color-primary)]">
                  <CheckCircle2 size={14} /> Completato
                </span>
              )}
            </h3>
            {place.address && (
              <p className="text-sm text-[var(--color-text-secondary)] truncate">
                {place.address}
              </p>
            )}
          </div>
          {place.category && (
            <span className="inline-block px-3 py-1 bg-[var(--color-bg-accent)] rounded-full text-sm whitespace-nowrap">
              {place.category.replace(/_/g, ' ')}
            </span>
          )}
        </div>
        {explanation && (
          <p className="mt-2 text-sm text-[var(--color-text-secondary)] line-clamp-2">
            ✨ {renderMarkdownBold(explanation)}
          </p>
        )}
      </div>
    </button>
  );
}
