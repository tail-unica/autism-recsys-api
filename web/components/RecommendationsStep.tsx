import { useState, useEffect } from 'react';
import { Loader2 } from 'lucide-react';
import { FeedbackModal } from './FeedbackModal';
import { getRecommendations } from '../lib/api';
import { Place, Recommendation } from '../lib/types';
import { apiConfig } from '../resources/api_config';

interface RecommendationsStepProps {
  userId: string;
  nickname?: string;
  profileAnswers: Record<string, number>;
  favoritePlaces: Place[];
  onEndSession: () => void;
}

export function RecommendationsStep({
  userId,
  nickname,
  profileAnswers,
  favoritePlaces,
  onEndSession,
}: RecommendationsStepProps) {
  const [loading, setLoading] = useState(true);
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [activeFeedbackIndex, setActiveFeedbackIndex] = useState<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    const fetchRecommendations = async () => {
      setLoading(true);
      try {
        const { items } = await getRecommendations({
          userId: userId || 'guest',
          profileAnswers,
          favoritePlaces,
        });
        if (!cancelled) setRecommendations(items);
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    fetchRecommendations();
    return () => {
      cancelled = true;
    };
  }, [userId, profileAnswers, favoritePlaces]);

  const handleFeedbackSubmit = (
    placeId: string,
    feedback: { liked: boolean; answers: Record<string, number>; comment: string }
  ) => {
    // TODO: Save feedback to Supabase
    console.log('Feedback submitted:', { placeId, ...feedback });

    // Save to localStorage for now
    const allFeedbacks = JSON.parse(localStorage.getItem('feedbacks') || '[]');
    allFeedbacks.push({
      userId,
      placeId,
      ...feedback,
      timestamp: new Date().toISOString(),
    });
    localStorage.setItem('feedbacks', JSON.stringify(allFeedbacks));
  };

  const activeRecommendation =
    activeFeedbackIndex !== null ? recommendations[activeFeedbackIndex] : null;

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="mx-auto mb-4 animate-spin text-[var(--color-primary)]" size={48} />
          <h2>Stiamo cercando i posti migliori per te...</h2>
          <p className="mt-2">Un momento di pazienza</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen p-6 py-12">
      <div className="max-w-6xl mx-auto">
        <div className="mb-8">
          <h1 className="mb-2">Raccomandazioni{nickname ? ` per ${nickname}` : ''}</h1>
          <p>Abbiamo selezionato questi posti in base alle tue preferenze</p>
        </div>

        {recommendations.length === 0 ? (
          <div className="p-8 bg-[var(--color-bg-secondary)] rounded-2xl text-center">
            Nessuna raccomandazione disponibile al momento.
          </div>
        ) : (
          <div className="space-y-4 mb-8">
            {recommendations.map((recommendation, index) => {
              const imageSrc = recommendation.image || apiConfig.fallback.placeholderImage;
              return (
                <button
                  key={recommendation.id}
                  type="button"
                  onClick={() => setActiveFeedbackIndex(index)}
                  className="w-full text-left bg-[var(--color-bg-secondary)] hover:bg-[var(--color-bg-accent)] transition-colors rounded-2xl p-4 md:p-5 shadow-sm"
                >
                  <div className="flex items-center gap-4">
                    <div className="w-20 h-20 rounded-xl overflow-hidden flex-shrink-0 bg-[var(--color-bg-accent)]">
                      <img
                        src={imageSrc}
                        alt={recommendation.name}
                        className="w-full h-full object-cover"
                        loading="lazy"
                      />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-start justify-between gap-4">
                        <div className="min-w-0">
                          <h3 className="mb-1 truncate">{recommendation.name}</h3>
                          {recommendation.address && (
                            <p className="text-sm text-[var(--color-text-secondary)] truncate">
                              {recommendation.address}
                            </p>
                          )}
                        </div>
                        {recommendation.category && (
                          <span className="inline-flex px-3 py-1 bg-white rounded-full text-sm whitespace-nowrap">
                            {recommendation.category}
                          </span>
                        )}
                      </div>
                      <p className="mt-2 text-sm text-[var(--color-text-secondary)] line-clamp-2">
                        ✨ {recommendation.explanation}
                      </p>
                    </div>
                  </div>
                </button>
              );
            })}
          </div>
        )}

        <div className="text-center">
          <button
            onClick={onEndSession}
            className="bg-[var(--color-primary)] hover:bg-[var(--color-primary-hover)] text-white py-4 px-8 rounded-xl transition-colors"
          >
            Termina Sessione
          </button>
        </div>
      </div>

      {activeRecommendation && activeFeedbackIndex !== null && (
        <FeedbackModal
          recommendation={activeRecommendation}
          hasNext={activeFeedbackIndex < recommendations.length - 1}
          onNext={() => {
            setActiveFeedbackIndex((prev) => {
              if (prev === null) return prev;
              const next = prev + 1;
              return next < recommendations.length ? next : null;
            });
          }}
          onClose={() => setActiveFeedbackIndex(null)}
          onSubmit={(feedback) => handleFeedbackSubmit(activeRecommendation.id, feedback)}
        />
      )}
    </div>
  );
}
