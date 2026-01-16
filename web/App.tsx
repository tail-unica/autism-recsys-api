import { useEffect, useState } from 'react';
import { NicknameStep } from './components/NicknameStep';
import { ProfileConfigStep } from './components/ProfileConfigStep';
import { FavoritePlacesStep } from './components/FavoritePlacesStep';
import { RecommendationsStep } from './components/RecommendationsStep';
import StepProgress, { type StepDefinition } from './components/StepProgress';
import { Place } from './lib/types';

type Step = 'nickname' | 'profile' | 'favorites' | 'recommendations';

export default function App() {
  const [currentStep, setCurrentStep] = useState<Step>('nickname');
  const [nickname, setNickname] = useState('');
  const [nicknameHash, setNicknameHash] = useState('');
  const [isNewUser, setIsNewUser] = useState(false);
  const [profileAnswers, setProfileAnswers] = useState<Record<string, number>>({});
  const [favoritePlaces, setFavoritePlaces] = useState<Place[]>([]);

  useEffect(() => {
    // When switching steps, start the page from the top.
    window.scrollTo({ top: 0, left: 0, behavior: 'auto' });
  }, [currentStep]);

  const steps: readonly StepDefinition<Step>[] = [
    { id: 'nickname', label: 'Nickname' },
    { id: 'profile', label: 'Profilo' },
    { id: 'favorites', label: 'Luoghi preferiti' },
    { id: 'recommendations', label: 'Raccomandazioni' },
  ] as const;

  const handleNicknameComplete = (nick: string, nickHash: string, isNew: boolean) => {
    // Keep nickname only as volatile UI value.
    setNickname(nick);
    setNicknameHash(nickHash);
    setIsNewUser(isNew);

    // If the user identity changes, don't carry over previous session state.
    setProfileAnswers({});
    setFavoritePlaces([]);

    // If returning user, check if they have a profile
    if (!isNew) {
      const users = JSON.parse(localStorage.getItem('users') || '{}');
      if (users[nickHash]?.profile) {
        // Skip profile step if already configured
        setProfileAnswers(users[nickHash].profile);
        setCurrentStep('favorites');
      } else {
        setCurrentStep('profile');
      }
    } else {
      setCurrentStep('profile');
    }
  };

  const handleProfileComplete = (answers: Record<string, number>) => {
    setProfileAnswers(answers);

    // Save profile to localStorage (will be moved to Supabase)
    const users = JSON.parse(localStorage.getItem('users') || '{}');
    if (users[nicknameHash]) {
      users[nicknameHash].profile = answers;
      localStorage.setItem('users', JSON.stringify(users));
    }

    setCurrentStep('favorites');
  };

  const handleFavoritesComplete = (places: Place[]) => {
    setFavoritePlaces(places);
    setCurrentStep('recommendations');
  };

  const canNavigateTo = (step: Step) => {
    switch (step) {
      case 'nickname':
        return true;
      case 'profile':
        return nicknameHash.trim().length > 0;
      case 'favorites':
        return nicknameHash.trim().length > 0 && Object.keys(profileAnswers).length > 0;
      case 'recommendations':
        return (
          nicknameHash.trim().length > 0 &&
          Object.keys(profileAnswers).length > 0 &&
          favoritePlaces.length > 0
        );
      default:
        return false;
    }
  };

  const handleNavigate = (step: Step) => {
    if (!canNavigateTo(step)) return;
    setCurrentStep(step);
  };

  const handleEndSession = () => {
    // Reset to start
    setCurrentStep('nickname');
    setNickname('');
    setNicknameHash('');
    setIsNewUser(false);
    setProfileAnswers({});
    setFavoritePlaces([]);
  };

  return (
    <div className="min-h-screen">
      <StepProgress
        steps={steps}
        currentStep={currentStep}
        canNavigateTo={canNavigateTo}
        onNavigate={handleNavigate}
      />

      {currentStep === 'nickname' && (
        <NicknameStep initialNickname={nickname} onComplete={handleNicknameComplete} />
      )}
      {currentStep === 'profile' && (
        <ProfileConfigStep
          userId={nicknameHash}
          nickname={nickname}
          initialAnswers={profileAnswers}
          onComplete={handleProfileComplete}
        />
      )}
      {currentStep === 'favorites' && (
        <FavoritePlacesStep initialPlaces={favoritePlaces} onComplete={handleFavoritesComplete} />
      )}
      {currentStep === 'recommendations' && (
        <RecommendationsStep
          userId={nicknameHash}
          nickname={nickname}
          profileAnswers={profileAnswers}
          favoritePlaces={favoritePlaces}
          onEndSession={handleEndSession}
        />
      )}
    </div>
  );
}
