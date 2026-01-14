import { useState } from 'react';
import { X } from 'lucide-react';
import { questions as surveyQuestions } from '../resources/survey';

interface FeedbackModalProps {
  placeName: string;
  liked: boolean;
  onClose: () => void;
  onSubmit: (feedback: { answers: Record<string, number>; comment: string }) => void;
}

export function FeedbackModal({ placeName, liked, onClose, onSubmit }: FeedbackModalProps) {
  const [answers, setAnswers] = useState<Record<string, number>>({});
  const [comment, setComment] = useState('');
  const totalQuestions = surveyQuestions.length;

  const handleAnswerChange = (questionId: string, value: number) => {
    setAnswers((prev) => ({ ...prev, [questionId]: value }));
  };

  const allAnswered = Object.keys(answers).length === totalQuestions;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!allAnswered) {
      alert('Rispondi a tutte le domande prima di inviare il feedback');
      return;
    }

    onSubmit({ answers, comment });
    onClose();
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-40 flex items-center justify-center p-6 z-50">
      <div className="bg-[var(--color-bg-secondary)] rounded-3xl p-10 max-w-4xl w-full relative shadow-2xl max-h-[90vh] overflow-y-auto">
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)]"
        >
          <X size={24} />
        </button>

        <h2 className="mb-4">Dicci di più sulla spiegazione</h2>
        <p className="mb-6">
          {placeName} - {liked ? 'Ti è piaciuto' : 'Non ti è piaciuto'}
        </p>

        <form onSubmit={handleSubmit} className="space-y-8">
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

          <div>
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
              disabled={!allAnswered}
              className={`flex-1 py-3 px-6 rounded-xl transition-all ${
                allAnswered
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
