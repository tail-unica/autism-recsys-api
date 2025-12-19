export type SensoryFeatureKey = 'light' | 'space' | 'crowd' | 'noise' | 'odor';

export type SensoryFeatures = Partial<Record<SensoryFeatureKey, number>>;

export interface Place {
  id: string;
  name: string;
  address?: string;
  category?: string;
  image?: string;
  description?: string;
  coordinates?: [number, number];
  sensory_features?: SensoryFeatures;
}

export interface Recommendation extends Place {
  explanation: string;
  score?: number;
}
