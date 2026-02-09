import { useState } from 'react';
import { questions as aversionQuestions } from '../resources/aversions_questions';
import { questions as profileQuestions } from '../resources/profile_questions';
import { updateUserProfile } from '../lib/backend';

interface ProfileConfigStepProps {
  userId: string;
  nickname?: string;
  initialAnswers?: Record<string, number>;
  onComplete: (answers: Record<string, number>) => void;
}

type ProfileQuestion = (typeof profileQuestions)[number];
type AversionQuestion = (typeof aversionQuestions)[number];

const totalQuestions = profileQuestions.length + aversionQuestions.length;

export function ProfileConfigStep({ userId: _userId, nickname, initialAnswers = {}, onComplete }: ProfileConfigStepProps) {
  const [answers, setAnswers] = useState<Record<string, number>>(initialAnswers);
  const [isSaving, setIsSaving] = useState(false);
  const [saveError, setSaveError] = useState('');

  const handleAnswerChange = (questionId: string, value: number | null) => {
    setAnswers((prev) => {
      if (value === null) {
        const { [questionId]: _removed, ...rest } = prev;
        return rest;
      }

      return { ...prev, [questionId]: value };
    });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaveError('');

    if (Object.keys(answers).length < totalQuestions) {
      alert('Rispondi a tutte le domande per continuare');
      return;
    }

    setIsSaving(true);

    try {
      // Salva il profilo sul backend
      await updateUserProfile(answers);
      onComplete(answers);
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : 'Errore nel salvataggio del profilo');
    } finally {
      setIsSaving(false);
    }
  };

  const allAnswered = Object.keys(answers).length === totalQuestions;

  const renderProfileQuestion = (question: ProfileQuestion, displayIndex: number) => (
    <div key={question.id} className="space-y-4">
      <label className="block">
        {displayIndex}. {question.text}
      </label>

      <div className="space-y-3">
        {question.type === 'number' ? (
          <div className="space-y-3">
            <input
              type="number"
              min={Number(question.min_label)}
              max={Number(question.max_label)}
              value={answers[question.id] ?? ''}
              onChange={(event) => {
                const { value } = event.target;
                if (value === '') {
                  handleAnswerChange(question.id, null);
                  return;
                }
                handleAnswerChange(question.id, Number(value));
              }}
              className="w-full py-3 px-4 rounded-xl border-2 border-[var(--color-border)] bg-white focus:border-[var(--color-primary)] focus:outline-none"
            />
            <div className="flex justify-between text-sm text-[var(--color-text-secondary)] px-1">
              <span>{question.min_label}</span>
              <span>{question.max_label}</span>
            </div>
          </div>
        ) : (
          <div className="space-y-3">
            <div className="flex gap-3 justify-between">
              {[
                { label: question.min_label, value: 0 },
                { label: question.max_label, value: 1 }
              ].map((option) => (
                <button
                  key={option.label}
                  type="button"
                  onClick={() => handleAnswerChange(question.id, option.value)}
                  className={`flex-1 py-3 px-4 rounded-xl border-2 transition-all ${
                    answers[question.id] === option.value
                      ? 'border-[var(--color-primary)] bg-[var(--color-bg-accent)] scale-105'
                      : 'border-[var(--color-border)] bg-white hover:border-[var(--color-primary)] hover:bg-[var(--color-bg-accent)]'
                  }`}
                >
                  {option.label}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );

  const renderAversionQuestion = (question: AversionQuestion, displayIndex: number) => (
    <div key={question.id} className="space-y-4">
      <label className="block">
        {displayIndex}. {question.text}
      </label>

      <div className="space-y-3">
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
  );

  return (
    <div className="min-h-screen p-6 py-12">
      <div className="max-w-3xl mx-auto">
        <div className="bg-[var(--color-bg-secondary)] rounded-3xl shadow-sm p-8 md:p-12">
          <h1 className="mb-2">Ciao{nickname ? `, ${nickname}` : ''}!</h1>
          <p className="mb-8">Rispondi a queste domande per aiutarci a conoscere le tue preferenze</p>

          <form onSubmit={handleSubmit} className="space-y-12">
            <section className="space-y-6">
              <div>
                <p className="text-sm uppercase tracking-widest text-[var(--color-text-secondary)]">Sezione 1</p>
                <h2 className="text-xl font-semibold">Informazioni personali</h2>
              </div>
              <div className="space-y-8">
                {profileQuestions.map((question, index) =>
                  renderProfileQuestion(question, index + 1)
                )}
              </div>
            </section>

            <section className="space-y-6 border-t border-[var(--color-border)] pt-10">
              <div>
                <p className="text-sm uppercase tracking-widest text-[var(--color-text-secondary)]">Sezione 2</p>
                <h2 className="text-xl font-semibold">Sensibilità e avversioni</h2>
              </div>
              <div className="space-y-8">
                {aversionQuestions.map((question, index) =>
                  renderAversionQuestion(question, index + 1)
                )}
              </div>
            </section>

            {saveError && (
              <p className="text-[var(--color-error)] text-center">{saveError}</p>
            )}

            <button
              type="submit"
              disabled={!allAnswered || isSaving}
              className={`w-full py-3 px-6 rounded-xl transition-all ${
                allAnswered && !isSaving
                  ? 'bg-[var(--color-primary)] hover:bg-[var(--color-primary-hover)] text-white'
                  : 'bg-[var(--color-border)] text-[var(--color-text-secondary)] cursor-not-allowed'
              }`}
            >
              {isSaving ? 'Salvataggio in corso...' : 'Continua'}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}