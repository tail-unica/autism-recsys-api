import { useLayoutEffect, useRef, useState } from 'react';
import { X } from 'lucide-react';
import { questions as surveyQuestions } from '../resources/survey';
import { Recommendation, SensoryFeatureKey } from '../lib/types';

import { submitFeedback } from '../lib/backend';
import { renderMarkdownBold } from '../lib/markdown';

interface FeedbackModalProps {
  onClose: (partial: { liked: boolean | null; answers: Record<string, number>; comment: string }) => void;
  onSubmit: (feedback: { liked: boolean; answers: Record<string, number>; comment: string }) => void;
  onNext: () => void;
  hasNext: boolean;
  recommendation: Recommendation;
  sessionId: string;
  isCompleted?: boolean;
  initialState?: { liked: boolean | null; answers: Record<string, number>; comment: string };
}

const SENSORY_LABELS: Record<SensoryFeatureKey, string> = {
  noise: 'Rumore',
  crowd: 'Affollamento',
  light: 'Intensità di Luce',
  space: 'Spazio',
  odor: 'Odori',
};

const SENSORY_EMOJI: Record<SensoryFeatureKey, string> = {
  noise: '🔊',
  crowd: '👥',
  light: '💡',
  space: '📏',
  odor: '👃',
};

const getSensoryLevel = (value: number): string => {
  if (value <= 1) return 'molto basso';
  if (value <= 2) return 'basso';
  if (value <= 3) return 'moderato';
  if (value <= 4) return 'alto';
  return 'molto alto';
};

export function FeedbackModal({ recommendation, sessionId, isCompleted, initialState, onClose, onSubmit, onNext, hasNext }: FeedbackModalProps) {
  const scrollContainerRef = useRef<HTMLDivElement | null>(null);
  const [liked, setLiked] = useState<boolean | null>(initialState?.liked ?? null);
  const [answers, setAnswers] = useState<Record<string, number>>(initialState?.answers ?? {});
  const [comment, setComment] = useState(initialState?.comment ?? '');
  const [touchedClose, setTouchedClose] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const totalQuestions = surveyQuestions.length;

  const scrollToTop = (behavior: ScrollBehavior = 'auto') => {
    scrollContainerRef.current?.scrollTo({ top: 0, behavior });
  };

  useLayoutEffect(() => {
    scrollToTop('auto');
  }, [recommendation.id]);

  const handleAnswerChange = (questionId: string, value: number) => {
    setAnswers((prev) => ({ ...prev, [questionId]: value }));
  };

  const allAnswered = liked !== null && Object.keys(answers).length === totalQuestions;



  // Salva il feedback sul backend
  const saveFeedback = async () => {
    if (!allAnswered) return false;
    
    setIsSaving(true);
    try {
      await submitFeedback({
        sessionId,
        placeId: recommendation.id,
        placeName: recommendation.name,
        liked: liked as boolean,
        answers,
        comment,
      });
      return true;
    } catch (error) {
      console.error('Errore nel salvataggio del feedback:', error);
      // Continua comunque, il feedback è stato raccolto
      return true;
    } finally {
      setIsSaving(false);
    }
  };

  const handleClose = async () => {
    if (!allAnswered) {
      setTouchedClose(true);
      return;
    }
    await saveFeedback();
    onSubmit({ liked: liked as boolean, answers, comment });
    onClose({ liked, answers, comment });
  };

  const handleGoNext = async () => {
    if (!allAnswered) {
      setTouchedClose(true);
      return;
    }
    await saveFeedback();
    onSubmit({ liked: liked as boolean, answers, comment });
    onNext();
    setLiked(null);
    setAnswers({});
    setComment('');
    setTouchedClose(false);
  };

  const showMustCompleteMessage = touchedClose && !allAnswered;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-40 flex items-center justify-center p-6 z-50">
      <div
        ref={scrollContainerRef}
        className="bg-[var(--color-bg-secondary)] rounded-3xl p-6 md:p-10 max-w-5xl w-full relative shadow-2xl max-h-[90vh] overflow-y-auto"
      >
        <button
          onClick={() => onClose({ liked, answers, comment })}
          className="absolute top-4 right-4 transition-colors text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)]"
        >
          <X size={24} />
        </button>

        <div className="space-y-8">
          {/* Place Details */}
          <div className="bg-[var(--color-bg-primary)] rounded-2xl overflow-hidden">
            <div className="p-6">
              <div className="flex items-start justify-between gap-4 mb-3">
                <div className="min-w-0">
                  <h2 className="mb-1">{recommendation.name}</h2>
                  {recommendation.address && (
                    <p className="text-sm text-[var(--color-text-secondary)]">{recommendation.address}</p>
                  )}
                </div>
                {recommendation.category && (
                  <span className="inline-block px-3 py-1 bg-[var(--color-bg-accent)] rounded-full text-sm whitespace-nowrap">
                    {recommendation.category.replace(/_/g, ' ')}
                  </span>
                )}
              </div>

              {recommendation.description && (
                <p className="text-sm text-[var(--color-text-secondary)] mb-4">{recommendation.description}</p>
              )}

              {recommendation.sensory_features && Object.keys(recommendation.sensory_features).length > 0 && (
                <div className="mb-4 p-4 bg-[var(--color-bg-accent)] rounded-xl">
                  <p className="mb-3">Caratteristiche Sensoriali</p>
                  <div className="space-y-2">
                    {Object.entries(recommendation.sensory_features).map(([key, value]) => (
                      <div key={key} className="flex gap-3 items-stretch">
                        <div
                          className="w-9 shrink-0 flex items-center justify-center text-xl"
                          aria-hidden="true"
                        >
                          {SENSORY_EMOJI[key as SensoryFeatureKey]}
                        </div>

                        <div className="flex-1 min-w-0">
                          <div className="flex justify-between text-sm mb-1">
                            <span className="truncate">{SENSORY_LABELS[key as SensoryFeatureKey] || key} ({getSensoryLevel(value as number)})</span>
                            <span>{value}/5</span>
                          </div>
                          <div className="h-2 bg-white rounded-full overflow-hidden">
                            <div
                              className="h-full bg-[var(--color-primary)] rounded-full transition-all"
                              style={{ width: `${((value as number) / 5) * 100}%` }}
                            />
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <div className="p-4 bg-[var(--color-bg-secondary)] rounded-xl">
                <p className="text-base leading-relaxed">{renderMarkdownBold(recommendation.explanation)}</p>
              </div>
            </div>
          </div>

          {/* Like / Dislike */}
          <div className="bg-[var(--color-bg-primary)] rounded-2xl p-6">
            <h3 className="mb-3">Ti è piaciuta questa spiegazione?</h3>
            <div className="flex gap-3">
              <button
                type="button"
                onClick={() => setLiked(true)}
                className={`flex-1 py-3 px-4 rounded-xl border-2 transition-all ${
                  liked === true
                    ? 'border-[var(--color-primary)] bg-[var(--color-bg-accent)]'
                    : 'border-[var(--color-border)] bg-white hover:border-[var(--color-primary)] hover:bg-[var(--color-bg-accent)]'
                }`}
              >
                Mi è piaciuto
              </button>
              <button
                type="button"
                onClick={() => setLiked(false)}
                className={`flex-1 py-3 px-4 rounded-xl border-2 transition-all ${
                  liked === false
                    ? 'border-[var(--color-primary)] bg-[var(--color-bg-accent)]'
                    : 'border-[var(--color-border)] bg-white hover:border-[var(--color-primary)] hover:bg-[var(--color-bg-accent)]'
                }`}
              >
                Non mi è piaciuto
              </button>
            </div>
            {showMustCompleteMessage && liked === null && (
              <p className="mt-2 text-sm text-[var(--color-error)]">Seleziona una risposta prima di continuare.</p>
            )}
          </div>

          {/* Survey */}
          <div className="bg-[var(--color-bg-primary)] rounded-2xl p-6">
            <h3 className="mb-4">Questionario sulla spiegazione</h3>
            <div className="space-y-6">
              {surveyQuestions.map((question, index) => (
                <div key={question.id} className="space-y-3">
                  <label className="block font-medium">
                    {index + 1}. {question.text}
                  </label>
                  <div className="space-y-2">
                    <div className="flex gap-3 justify-between">
                      {[1, 2, 3, 4, 5].map((value) => (
                        <button
                          key={value}
                          type="button"
                          onClick={() => handleAnswerChange(question.id, value)}
                          className={`flex-1 py-3 px-4 rounded-xl border-2 transition-all ${
                            answers[question.id] === value
                              ? 'border-[var(--color-primary)] bg-[var(--color-bg-accent)] scale-105'
                              : 'border-[var(--color-border)] bg-white hover:border-[var(--color-primary)] hover:bg-[var(--color-bg-accent)]'
                          }`}
                        >
                          {value}
                        </button>
                      ))}
                    </div>
                    <div className="flex justify-between text-sm text-[var(--color-text-secondary)] px-1">
                      <span>{question.min_label}</span>
                      <span>{question.max_label}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            <div className="mt-8">
              <label htmlFor="comment" className="block mb-2">
                Commento (opzionale)
              </label>
              <textarea
                id="comment"
                value={comment}
                onChange={(e) => setComment(e.target.value)}
                rows={4}
                placeholder="Cosa ti è piaciuto o non ti è piaciuto di questa spiegazione?"
                className="w-full px-4 py-3 rounded-xl border-2 border-[var(--color-border)] bg-white focus:border-[var(--color-primary)] focus:outline-none transition-colors resize-none"
              />
            </div>

            {showMustCompleteMessage && !allAnswered && (
              <p className="mt-3 text-sm text-[var(--color-error)]">
                Completa tutte le domande (e la preferenza) per poter chiudere o passare al prossimo.
              </p>
            )}
          </div>

          <div className="space-y-6">
            <div className="flex gap-3">
              <button
                type="button"
                onClick={() => {
                  if (!allAnswered) setTouchedClose(true);
                  else handleClose();
                }}
                disabled={!allAnswered || isSaving}
                className={`flex-1 py-3 px-6 rounded-xl transition-all ${
                  allAnswered && !isSaving
                    ? 'bg-[var(--color-primary)] hover:bg-[var(--color-primary-hover)] text-white'
                    : 'bg-[var(--color-border)] text-[var(--color-text-secondary)] cursor-not-allowed'
                }`}
              >
                {isSaving ? 'Salvataggio...' : 'Chiudi'}
              </button>
              <button
                type="button"
                onClick={handleGoNext}
                disabled={!allAnswered || !hasNext || isSaving}
                className={`flex-1 py-3 px-6 rounded-xl transition-all ${
                  allAnswered && hasNext && !isSaving
                    ? 'border-2 border-[var(--color-primary)] text-[var(--color-primary)] hover:bg-[var(--color-bg-accent)]'
                    : 'border-2 border-[var(--color-border)] text-[var(--color-text-secondary)] cursor-not-allowed'
                }`}
              >
                {isSaving ? 'Salvataggio...' : 'Prossima'}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
