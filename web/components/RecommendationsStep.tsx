import { useState, useEffect } from 'react';
import { Loader2 } from 'lucide-react';
import { FeedbackModal } from './FeedbackModal';
import { requestRecommendations, RecommendationResponse } from '../lib/backend';
import { Place, Recommendation } from '../lib/types';
import { apiConfig } from '../resources/api_config';

interface RecommendationsStepProps {
  userId: string;
  nickname?: string;
  profileAnswers: Record<string, number>;
  favoritePlaces: Place[];
  onEndSession: () => void;
}

// Costruisce le avversioni dal profilo
const buildAversions = (profileAnswers: Record<string, number>) => {
  const aversions: Record<string, number> = { ...apiConfig.aversionDefaults };
  Object.entries(apiConfig.profileQuestionToAversion).forEach(([questionId, aversionKey]) => {
    const answer = profileAnswers[questionId];
    if (typeof answer === 'number') {
      aversions[aversionKey] = answer;
    }
  });
  // Aggiungi anche le risposte dirette alle avversioni
  const aversionKeys = ['bright_light', 'dim_light', 'crowd', 'noise', 'odor', 'narrow_space', 'wide_space'];
  aversionKeys.forEach(key => {
    if (typeof profileAnswers[key] === 'number') {
      aversions[key] = profileAnswers[key];
    }
  });
  return Object.entries(aversions).map(([feature_name, rating]) => ({ feature_name, rating }));
};

export function RecommendationsStep({
  userId,
  nickname,
  profileAnswers,
  favoritePlaces,
  onEndSession,
}: RecommendationsStepProps) {
  const [loading, setLoading] = useState(true);
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [sessionId, setSessionId] = useState<string>('');
  const [activeFeedbackIndex, setActiveFeedbackIndex] = useState<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    const fetchRecommendations = async () => {
      setLoading(true);
      try {
        const response: RecommendationResponse = await requestRecommendations({
          preferences: favoritePlaces.map(p => p.name),
          aversions: buildAversions(profileAnswers),
        });
        if (!cancelled) {
          setRecommendations(response.recommendations as Recommendation[]);
          setSessionId(response.sessionId);
        }
      } catch (error) {
        console.error('Errore nel recupero delle raccomandazioni:', error);
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    fetchRecommendations();
    return () => {
      cancelled = true;
    };
  }, [userId, profileAnswers, favoritePlaces]);

  const handleFeedbackSubmit = async (
    placeId: string,
    feedback: { liked: boolean; answers: Record<string, number>; comment: string }
  ) => {
    // Il salvataggio del feedback è gestito nel FeedbackModal
    console.log('Feedback submitted:', { placeId, sessionId, ...feedback });
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
          sessionId={sessionId}
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
