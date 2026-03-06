import { useState } from 'react';
import { MapPin, ThumbsUp, ThumbsDown, MessageSquare } from 'lucide-react';
import { ImageWithFallback } from './figma/ImageWithFallback';
import { Recommendation, SensoryFeatureKey } from '../lib/types';
import { apiConfig } from '../resources/api_config';
import { renderMarkdownBold } from '../lib/markdown';

interface PlaceCardProps {
  recommendation: Recommendation;
  onFeedback: (placeId: string, liked: boolean) => void;
}

const SENSORY_LABELS: Record<SensoryFeatureKey, string> = {
  noise: 'Rumore',
  crowd: 'Affollamento',
  light: 'Intensità Luce',
  space: 'Spazio',
  odor: 'Odori',
};

export function PlaceCard({ recommendation, onFeedback }: PlaceCardProps) {
  const [showFeedback, setShowFeedback] = useState(false);
  const [feedbackGiven, setFeedbackGiven] = useState<'liked' | 'disliked' | null>(null);
  const imageSrc = recommendation.image || apiConfig.fallback.placeholderImage;

  const handleFeedback = (liked: boolean) => {
    setFeedbackGiven(liked ? 'liked' : 'disliked');
    setShowFeedback(true);
    onFeedback(recommendation.id, liked);
  };

  return (
    <div className="bg-[var(--color-bg-secondary)] rounded-2xl overflow-hidden shadow-sm">
      {/* Image */}
      <div className="relative h-56 bg-[var(--color-bg-accent)]">
        <ImageWithFallback
          src={imageSrc}
          alt={recommendation.name}
          className="w-full h-full object-cover"
        />
      </div>

      {/* Content */}
      <div className="p-6">
        <div className="mb-4">
          <h3 className="mb-2">{recommendation.name}</h3>
          <div className="flex items-start gap-2 text-[var(--color-text-secondary)] mb-2">
            <MapPin size={16} className="mt-1 flex-shrink-0" />
            <span className="text-sm">{recommendation.address}</span>
          </div>
          {recommendation.category && (
            <span className="inline-block px-3 py-1 bg-[var(--color-bg-accent)] rounded-full text-sm">
              {recommendation.category.replace(/_/g, ' ')}
            </span>
          )}
        </div>

        {/* Sensory Features */}
        {recommendation.sensory_features && Object.keys(recommendation.sensory_features).length > 0 && (
          <div className="mb-4 p-4 bg-[var(--color-bg-accent)] rounded-xl">
            <p className="mb-3">Caratteristiche Sensoriali</p>
            <div className="space-y-2">
              {Object.entries(recommendation.sensory_features).map(([key, value]) => (
                <div key={key}>
                  <div className="flex justify-between text-sm mb-1">
                    <span>{SENSORY_LABELS[key as SensoryFeatureKey] || key}</span>
                    <span>{value}/5</span>
                  </div>
                  <div className="h-2 bg-white rounded-full overflow-hidden">
                    <div
                      className="h-full bg-[var(--color-primary)] rounded-full transition-all"
                      style={{ width: `${(value / 5) * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Explanation */}
        <div className="mb-6 p-4 bg-[var(--color-bg-primary)] rounded-xl">
          <div className="flex items-start gap-2">
            <MessageSquare size={16} className="mt-1 flex-shrink-0 text-[var(--color-primary)]" />
            <p className="text-base leading-relaxed">{renderMarkdownBold(recommendation.explanation)}</p>
          </div>
        </div>

        {/* Feedback Buttons */}
        {feedbackGiven === null ? (
          <div className="flex gap-3">
            <button
              onClick={() => handleFeedback(true)}
              className="flex-1 flex items-center justify-center gap-2 py-3 px-4 bg-[var(--color-success)] hover:opacity-80 text-white rounded-xl transition-all"
            >
              <ThumbsUp size={20} />
              Mi è piaciuto
            </button>
            <button
              onClick={() => handleFeedback(false)}
              className="flex-1 flex items-center justify-center gap-2 py-3 px-4 bg-[var(--color-error)] hover:opacity-80 text-white rounded-xl transition-all"
            >
              <ThumbsDown size={20} />
              Non mi è piaciuto
            </button>
          </div>
        ) : (
          <div
            className={`p-4 rounded-xl text-center text-white ${
              feedbackGiven === 'liked' ? 'bg-[var(--color-success)]' : 'bg-[var(--color-error)]'
            }`}
          >
            Grazie per il tuo feedback!
          </div>
        )}
      </div>
    </div>
  );
}
