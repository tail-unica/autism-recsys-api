import { useState, useEffect } from 'react';
import { Loader2 } from 'lucide-react';
import { PlaceCard } from './PlaceCard';
import { FeedbackModal } from './FeedbackModal';
import { getRecommendations } from '../lib/api';
import { Place, Recommendation } from '../lib/types';

interface RecommendationsStepProps {
  nickname: string;
  profileAnswers: Record<string, number>;
  favoritePlaces: Place[];
  onEndSession: () => void;
}

export function RecommendationsStep({
  nickname,
  profileAnswers,
  favoritePlaces,
  onEndSession,
}: RecommendationsStepProps) {
  const [loading, setLoading] = useState(true);
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [feedbacks, setFeedbacks] = useState<Record<string, { liked: boolean }>>({});
  const [activeFeedbackModal, setActiveFeedbackModal] = useState<{
    placeId: string;
    placeName: string;
    liked: boolean;
  } | null>(null);

  useEffect(() => {
    let cancelled = false;
    const fetchRecommendations = async () => {
      setLoading(true);
      try {
        const { items } = await getRecommendations({
          userId: nickname || 'guest',
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
  }, [nickname, profileAnswers, favoritePlaces]);

  const handleFeedback = (placeId: string, liked: boolean) => {
    setFeedbacks((prev) => ({ ...prev, [placeId]: { liked } }));
    const place = recommendations.find((r) => r.id === placeId);
    if (place) {
      setActiveFeedbackModal({ placeId, placeName: place.name, liked });
    }
  };

  const handleFeedbackSubmit = (
    placeId: string,
    feedback: { answers: Record<string, number>; comment: string }
  ) => {
    // TODO: Save feedback to Supabase
    console.log('Feedback submitted:', { placeId, ...feedback });

    // Save to localStorage for now
    const allFeedbacks = JSON.parse(localStorage.getItem('feedbacks') || '[]');
    allFeedbacks.push({
      nickname,
      placeId,
      ...feedback,
      timestamp: new Date().toISOString(),
    });
    localStorage.setItem('feedbacks', JSON.stringify(allFeedbacks));
  };

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
          <h1 className="mb-2">Raccomandazioni per {nickname}</h1>
          <p>Abbiamo selezionato questi posti in base alle tue preferenze</p>
        </div>

        {recommendations.length === 0 ? (
          <div className="p-8 bg-[var(--color-bg-secondary)] rounded-2xl text-center">
            Nessuna raccomandazione disponibile al momento.
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
            {recommendations.map((recommendation) => (
              <PlaceCard key={recommendation.id} recommendation={recommendation} onFeedback={handleFeedback} />
            ))}
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

      {activeFeedbackModal && (
        <FeedbackModal
          placeName={activeFeedbackModal.placeName}
          liked={activeFeedbackModal.liked}
          onClose={() => setActiveFeedbackModal(null)}
          onSubmit={(feedback) => handleFeedbackSubmit(activeFeedbackModal.placeId, feedback)}
        />
      )}
    </div>
  );
}
