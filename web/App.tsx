import { useEffect, useState } from 'react';
import { NicknameStep } from './components/NicknameStep';
import { ProfileConfigStep } from './components/ProfileConfigStep';
import { FavoritePlacesStep } from './components/FavoritePlacesStep';
import { RecommendationsStep } from './components/RecommendationsStep';
import StepProgress, { type StepDefinition } from './components/StepProgress';
import { Place } from './lib/types';
import { isAuthenticated, verifyToken, logout, clearToken } from './lib/auth';

type Step = 'nickname' | 'profile' | 'favorites' | 'recommendations';

export default function App() {
  const [currentStep, setCurrentStep] = useState<Step>('nickname');
  const [nickname, setNickname] = useState('');
  const [nicknameHash, setNicknameHash] = useState('');
  const [isNewUser, setIsNewUser] = useState(false);
  const [profileAnswers, setProfileAnswers] = useState<Record<string, number>>({});
  const [favoritePlaces, setFavoritePlaces] = useState<Place[]>([]);
  const [studyCodeFromLink, setStudyCodeFromLink] = useState('');
  const [isLoading, setIsLoading] = useState(true);

  // Verifica lo stato di autenticazione all'avvio
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const studyCode = (params.get('study') || params.get('secret') || '').trim();
    if (studyCode) {
      setStudyCodeFromLink(studyCode);
    }
  }, []);

  useEffect(() => {
    const checkAuth = async () => {
      if (isAuthenticated()) {
        try {
          const { valid, hasProfile } = await verifyToken();
          if (valid) {
            // Utente già autenticato, salta alla schermata appropriata
            if (hasProfile) {
              setCurrentStep('favorites');
            } else {
              setCurrentStep('profile');
            }
          } else {
            clearToken();
          }
        } catch {
          clearToken();
        }
      }
      setIsLoading(false);
    };

    checkAuth();
  }, []);

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

  const handleNicknameComplete = (nick: string, nickHash: string, isNew: boolean, profile?: Record<string, number>, favorites?: Place[]) => {
    // Keep nickname only as volatile UI value.
    setNickname(nick);
    setNicknameHash(nickHash);
    setIsNewUser(isNew);

    // Carica i dati dal backend se disponibili
    if (profile && Object.keys(profile).length > 0) {
      setProfileAnswers(profile);
    } else {
      setProfileAnswers({});
    }

    if (favorites && favorites.length > 0) {
      setFavoritePlaces(favorites);
    } else {
      setFavoritePlaces([]);
    }

    // Se l'utente ha già un profilo, salta alla selezione dei luoghi
    if (!isNew && profile && Object.keys(profile).length > 0) {
      setCurrentStep('favorites');
    } else {
      setCurrentStep('profile');
    }
  };

  const handleProfileComplete = (answers: Record<string, number>) => {
    setProfileAnswers(answers);
    // Il salvataggio su DB è gestito nel componente ProfileConfigStep
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

  const handleEndSession = async () => {
    // Logout e reset
    await logout();
    setCurrentStep('nickname');
    setNickname('');
    setNicknameHash('');
    setIsNewUser(false);
    setProfileAnswers({});
    setFavoritePlaces([]);
  };

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-[var(--color-primary)] mx-auto mb-4"></div>
          <p>Caricamento...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen">
      <StepProgress
        steps={steps}
        currentStep={currentStep}
        canNavigateTo={canNavigateTo}
        onNavigate={handleNavigate}
      />

      {currentStep === 'nickname' && (
        <NicknameStep
          initialNickname={nickname}
          initialStudyCode={studyCodeFromLink}
          onComplete={handleNicknameComplete}
        />
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
