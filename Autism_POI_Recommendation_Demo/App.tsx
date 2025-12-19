import { useState } from 'react';
import { NicknameStep } from './components/NicknameStep';
import { ProfileConfigStep } from './components/ProfileConfigStep';
import { FavoritePlacesStep } from './components/FavoritePlacesStep';
import { RecommendationsStep } from './components/RecommendationsStep';
import { Place } from './lib/types';

type Step = 'nickname' | 'profile' | 'favorites' | 'recommendations';

export default function App() {
  const [currentStep, setCurrentStep] = useState<Step>('nickname');
  const [nickname, setNickname] = useState('');
  const [isNewUser, setIsNewUser] = useState(false);
  const [profileAnswers, setProfileAnswers] = useState<Record<string, number>>({});
  const [favoritePlaces, setFavoritePlaces] = useState<Place[]>([]);

  const handleNicknameComplete = (nick: string, isNew: boolean) => {
    setNickname(nick);
    setIsNewUser(isNew);

    // If returning user, check if they have a profile
    if (!isNew) {
      const users = JSON.parse(localStorage.getItem('users') || '{}');
      if (users[nick]?.profile) {
        // Skip profile step if already configured
        setProfileAnswers(users[nick].profile);
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
    if (users[nickname]) {
      users[nickname].profile = answers;
      localStorage.setItem('users', JSON.stringify(users));
    }

    setCurrentStep('favorites');
  };

  const handleFavoritesComplete = (places: Place[]) => {
    setFavoritePlaces(places);
    setCurrentStep('recommendations');
  };

  const handleEndSession = () => {
    // Reset to start
    setCurrentStep('nickname');
    setNickname('');
    setIsNewUser(false);
    setProfileAnswers({});
    setFavoritePlaces([]);
  };

  return (
    <>
      {currentStep === 'nickname' && <NicknameStep onComplete={handleNicknameComplete} />}
      {currentStep === 'profile' && <ProfileConfigStep nickname={nickname} onComplete={handleProfileComplete} />}
      {currentStep === 'favorites' && <FavoritePlacesStep onComplete={handleFavoritesComplete} />}
      {currentStep === 'recommendations' && (
        <RecommendationsStep
          nickname={nickname}
          profileAnswers={profileAnswers}
          favoritePlaces={favoritePlaces}
          onEndSession={handleEndSession}
        />
      )}
    </>
  );
}
