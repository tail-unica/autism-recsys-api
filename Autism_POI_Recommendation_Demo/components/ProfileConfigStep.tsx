import { useState } from 'react';
import { questions } from '../resources/questions';

interface ProfileConfigStepProps {
  nickname: string;
  onComplete: (answers: Record<string, number>) => void;
}

export function ProfileConfigStep({ nickname, onComplete }: ProfileConfigStepProps) {
  const [answers, setAnswers] = useState<Record<string, number>>({});

  const handleAnswerChange = (questionId: string, value: number) => {
    setAnswers((prev) => ({ ...prev, [questionId]: value }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    if (Object.keys(answers).length < questions.length) {
      alert('Rispondi a tutte le domande per continuare');
      return;
    }

    onComplete(answers);
  };

  const allAnswered = Object.keys(answers).length === questions.length;

  return (
    <div className="min-h-screen p-6 py-12">
      <div className="max-w-3xl mx-auto">
        <div className="bg-[var(--color-bg-secondary)] rounded-3xl shadow-sm p-8 md:p-12">
          <h1 className="mb-2">Ciao, {nickname}!</h1>
          <p className="mb-8">Rispondi a queste domande per aiutarci a conoscere le tue preferenze</p>

          <form onSubmit={handleSubmit} className="space-y-8">
            {questions.map((question, index) => (
              <div key={question.id} className="space-y-4">
                <label className="block">
                  {index + 1}. {question.text}
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
            ))}

            <button
              type="submit"
              disabled={!allAnswered}
              className={`w-full py-3 px-6 rounded-xl transition-all ${
                allAnswered
                  ? 'bg-[var(--color-primary)] hover:bg-[var(--color-primary-hover)] text-white'
                  : 'bg-[var(--color-border)] text-[var(--color-text-secondary)] cursor-not-allowed'
              }`}
            >
              Continua
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}