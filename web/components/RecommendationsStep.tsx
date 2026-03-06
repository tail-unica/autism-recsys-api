import { useState, useEffect, useCallback } from 'react';
import { Loader2, CheckCircle2, ClipboardList } from 'lucide-react';
import { FeedbackModal } from './FeedbackModal';
import { requestRecommendations, RecommendationResponse } from '../lib/backend';
import { Place, Recommendation } from '../lib/types';
import { apiConfig, buildAversions } from '../resources/api_config';
import { renderMarkdownBold } from '../lib/markdown';

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
  const [sessionId, setSessionId] = useState<string>('');
  const [activeFeedbackIndex, setActiveFeedbackIndex] = useState<number | null>(null);
  const [completedFeedbacks, setCompletedFeedbacks] = useState<Set<string>>(new Set());
  // Stores partial answers so users can exit and re-enter
  const [savedFeedbacks, setSavedFeedbacks] = useState<
    Record<string, { liked: boolean | null; answers: Record<string, number>; comment: string }>
  >({});

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
    console.log('Feedback submitted:', { placeId, sessionId, ...feedback });
    setCompletedFeedbacks((prev) => new Set(prev).add(placeId));
    // Clear saved partial state once fully submitted
    setSavedFeedbacks((prev) => {
      const { [placeId]: _, ...rest } = prev;
      return rest;
    });
  };

  // Save partial state when user exits without completing
  const handleSavePartial = useCallback(
    (placeId: string, partial: { liked: boolean | null; answers: Record<string, number>; comment: string }) => {
      setSavedFeedbacks((prev) => ({ ...prev, [placeId]: partial }));
    },
    []
  );

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

  const completedCount = completedFeedbacks.size;
  const totalCount = recommendations.length;

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
          <>
            {/* Instructions */}
            <div className="flex items-start gap-3 mb-4 p-4 bg-[var(--color-bg-accent)] rounded-xl">
              <ClipboardList size={20} className="mt-0.5 flex-shrink-0 text-[var(--color-primary)]" />
              <p className="text-sm">
                Clicca su ogni scheda per visualizzare i dettagli del luogo e compilare il questionario di valutazione.
              </p>
            </div>

            {/* Progress bar */}
            <div className="mb-6">
              <div className="flex items-center justify-between text-sm mb-2">
                <span className="font-medium">Questionari completati</span>
                <span className="text-[var(--color-text-secondary)]">{completedCount}/{totalCount}</span>
              </div>
              <div className="h-2.5 bg-[var(--color-border)] rounded-full overflow-hidden">
                <div
                  className="h-full bg-[var(--color-primary)] rounded-full transition-all duration-500"
                  style={{ width: `${totalCount > 0 ? (completedCount / totalCount) * 100 : 0}%` }}
                />
              </div>
            </div>

            <div className="space-y-4 mb-8">
              {recommendations.map((recommendation, index) => {
                const imageSrc = recommendation.image || apiConfig.fallback.placeholderImage;
                const isCompleted = completedFeedbacks.has(recommendation.id);
                return (
                  <button
                    key={recommendation.id}
                    type="button"
                    onClick={() => setActiveFeedbackIndex(index)}
                    className={`w-full text-left transition-colors rounded-2xl p-4 md:p-5 shadow-sm ${
                      isCompleted
                        ? 'bg-[var(--color-bg-accent)] border-2 border-[var(--color-primary)]'
                        : 'bg-[var(--color-bg-secondary)] hover:bg-[var(--color-bg-accent)]'
                    }`}
                  >
                    <div className="flex items-center gap-4">
                      <div className="w-20 h-20 rounded-xl overflow-hidden flex-shrink-0 bg-[var(--color-bg-accent)] relative">
                        <img
                          src={imageSrc}
                          alt={recommendation.name}
                          className="w-full h-full object-cover"
                          loading="lazy"
                        />
                        {isCompleted && (
                          <div className="absolute inset-0 bg-black bg-opacity-40 flex items-center justify-center">
                            <CheckCircle2 size={28} className="text-white" />
                          </div>
                        )}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-start justify-between gap-4">
                          <div className="min-w-0">
                            <h3 className="mb-1 truncate flex items-center gap-2">
                              {recommendation.name}
                              {isCompleted && (
                                <span className="inline-flex items-center gap-1 text-xs font-medium text-[var(--color-primary)]">
                                  <CheckCircle2 size={14} /> Completato
                                </span>
                              )}
                            </h3>
                            {recommendation.address && (
                              <p className="text-sm text-[var(--color-text-secondary)] truncate">
                                {recommendation.address}
                              </p>
                            )}
                          </div>
                          {recommendation.category && (
                            <span className="inline-block px-3 py-1 bg-[var(--color-bg-accent)] rounded-full text-sm whitespace-nowrap">
                              {recommendation.category.replace(/_/g, ' ')}
                            </span>
                          )}
                        </div>
                        <p className="mt-2 text-sm text-[var(--color-text-secondary)] line-clamp-2">
                          ✨ {renderMarkdownBold(recommendation.explanation)}
                        </p>
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>
          </>
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
          isCompleted={completedFeedbacks.has(activeRecommendation.id)}
          initialState={savedFeedbacks[activeRecommendation.id]}
          hasNext={activeFeedbackIndex < recommendations.length - 1}
          onNext={() => {
            setActiveFeedbackIndex((prev) => {
              if (prev === null) return prev;
              const next = prev + 1;
              return next < recommendations.length ? next : null;
            });
          }}
          onClose={(partial) => {
            handleSavePartial(activeRecommendation.id, partial);
            setActiveFeedbackIndex(null);
          }}
          onSubmit={(feedback) => handleFeedbackSubmit(activeRecommendation.id, feedback)}
        />
      )}
    </div>
  );
}
