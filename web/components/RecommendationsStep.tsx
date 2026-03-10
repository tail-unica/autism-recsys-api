import { useState, useEffect, useCallback } from 'react';
import { Loader2, ClipboardList } from 'lucide-react';
import { FeedbackModal } from './FeedbackModal';
import { PlaceCard } from './PlaceCard';
import { requestRecommendations, RecommendationResponse } from '../lib/backend';
import { Place, Recommendation } from '../lib/types';
import { buildAversions } from '../resources/api_config';

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
              {recommendations.map((recommendation, index) => (
                <PlaceCard
                  key={recommendation.id}
                  place={recommendation}
                  explanation={recommendation.explanation}
                  isCompleted={completedFeedbacks.has(recommendation.id)}
                  onClick={() => setActiveFeedbackIndex(index)}
                />
              ))}
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
