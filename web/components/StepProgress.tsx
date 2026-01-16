import React from 'react';

export type StepDefinition<TStep extends string> = {
  id: TStep;
  label: string;
};

type StepProgressProps<TStep extends string> = {
  steps: readonly StepDefinition<TStep>[];
  currentStep: TStep;
  canNavigateTo: (step: TStep) => boolean;
  onNavigate: (step: TStep) => void;
};

export default function StepProgress<TStep extends string>({
  steps,
  currentStep,
  canNavigateTo,
  onNavigate,
}: StepProgressProps<TStep>) {
  const currentIndex = Math.max(
    0,
    steps.findIndex((s) => s.id === currentStep)
  );

  const progressFraction =
    steps.length <= 1 ? 0 : currentIndex / (steps.length - 1);
  const clampedProgressFraction = Math.min(1, Math.max(0, progressFraction));

  // Tailwind `gap-2` = 0.5rem = 8px
  const stepGapPx = 8;
  const totalGapPx = stepGapPx * Math.max(0, steps.length - 1);

  // Each step item is `flex-1` and the circle is centered inside it.
  // The centers of first/last circles are half of the first/last item width from the edges.
  const colWidthExpr = steps.length > 0
    ? `(100% - ${totalGapPx}px) / ${steps.length}`
    : '0px';

  // Fill starts from the very left, and reaches:
  // - always up to the first circle center (colWidth/2)
  // - plus a percentage of the distance between first and last centers (100% - colWidth)
  const filledWidth =
    steps.length <= 1
      ? '0%'
      : `calc((${colWidthExpr}) / 2 + (100% - (${colWidthExpr})) * ${clampedProgressFraction})`;

  const circleRadiusPx = 18; // h-9 / 2 (used for vertical alignment)

  const CheckIcon = ({ className }: { className?: string }) => (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="3"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden="true"
    >
      <path d="M20 6L9 17l-5-5" />
    </svg>
  );

  return (
    <div className="sticky top-0 z-20 border-b border-[var(--color-border)] bg-[var(--color-bg-primary)]">
      <div className="max-w-4xl mx-auto px-6 py-4">
        <nav aria-label="Progresso">
          <ol className="relative flex items-start justify-between gap-2">
            <div
              aria-hidden="true"
              className="pointer-events-none absolute left-0 right-0 top-[18px] h-[3px] -translate-y-1/2 rounded-full bg-[var(--color-border)] z-0"
            />
            <div
              aria-hidden="true"
              className="pointer-events-none absolute left-0 top-[18px] h-[3px] -translate-y-1/2 rounded-full bg-[var(--color-primary)] transition-[width] duration-300 z-0"
              style={{ width: filledWidth, maxWidth: '100%' }}
            />

            {steps.map((step, index) => {
              const isActive = step.id === currentStep;
              const isEnabled = canNavigateTo(step.id);
              const isCompleted = index < currentIndex;

              const circleBase =
                'h-9 w-9 rounded-full flex items-center justify-center border-2 transition-colors';
              const circleClass = isCompleted || isActive
                ? `${circleBase} bg-[var(--color-primary)] border-[var(--color-primary)] text-white`
                : `${circleBase} bg-[var(--color-border)] border-[var(--color-border)] text-[var(--color-text-secondary)]`;

              const labelClass =
                'mt-3 text-xs font-semibold tracking-widest uppercase ' +
                (isCompleted || isActive
                  ? 'text-[var(--color-text-primary)]'
                  : 'text-[var(--color-text-secondary)]');

              return (
                <li key={step.id} className="flex-1">
                  <button
                    type="button"
                    onClick={() => onNavigate(step.id)}
                    disabled={!isEnabled}
                    aria-current={isActive ? 'step' : undefined}
                    className={
                      'relative z-10 w-full flex flex-col items-center text-center ' +
                      (!isEnabled ? 'cursor-not-allowed opacity-70' : 'hover:opacity-90')
                    }
                  >
                    <span className={`${circleClass} relative z-10`}>
                      {isCompleted ? (
                        <CheckIcon className="h-5 w-5" />
                      ) : (
                        <span className="text-sm font-semibold">{index + 1}</span>
                      )}
                    </span>
                    <span className={labelClass}>{step.label}</span>
                  </button>
                </li>
              );
            })}
          </ol>
        </nav>
      </div>
    </div>
  );
}
