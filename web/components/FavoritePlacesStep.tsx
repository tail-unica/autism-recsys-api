import { useEffect, useMemo, useState } from 'react';
import { Search, X, Loader2 } from 'lucide-react';
import { searchPlaces, availableCategories } from '../lib/api';
import { apiConfig } from '../resources/api_config';
import { Place } from '../lib/types';

interface FavoritePlacesStepProps {
  onComplete: (favoritePlaces: Place[]) => void;
}

const ALL_CATEGORY = 'Tutti';

export function FavoritePlacesStep({ onComplete }: FavoritePlacesStepProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState(ALL_CATEGORY);
  const [loading, setLoading] = useState(true);
  const [places, setPlaces] = useState<Place[]>([]);
  const [selectedPlaces, setSelectedPlaces] = useState<Place[]>([]);

  const categoryOptions = useMemo(
    () => [{ id: ALL_CATEGORY, name: ALL_CATEGORY }, ...availableCategories()],
    []
  );

  useEffect(() => {
    let cancelled = false;
    const timer = setTimeout(async () => {
      setLoading(true);
      try {
        const categoryFilter = selectedCategory === ALL_CATEGORY ? undefined : [selectedCategory];
        const { places: fetched } = await searchPlaces({ query: searchQuery, categories: categoryFilter });
        if (!cancelled) setPlaces(fetched);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }, 250);

    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [searchQuery, selectedCategory]);

  const togglePlace = (place: Place) => {
    setSelectedPlaces((prev) =>
      prev.find((p) => p.id === place.id) ? prev.filter((p) => p.id !== place.id) : [...prev, place]
    );
  };

  const removePlace = (placeId: string) => {
    setSelectedPlaces((prev) => prev.filter((p) => p.id !== placeId));
  };

  const handleContinue = () => {
    if (selectedPlaces.length === 0) return;
    onComplete(selectedPlaces);
  };

  return (
    <div className="min-h-screen p-6 py-12">
      <div className="max-w-5xl mx-auto">
        <div className="bg-[var(--color-bg-secondary)] rounded-3xl shadow-sm p-8 md:p-12">
          <h1 className="mb-2">Luoghi che ti piacciono</h1>
          <p className="mb-8">Seleziona alcuni luoghi che hai già visitato e che ti sono piaciuti</p>

          {/* Search Bar */}
          <div className="mb-6">
            <div className="relative">
              <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-[var(--color-text-secondary)]" size={20} />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Cerca un luogo..."
                className="w-full pl-12 pr-4 py-3 rounded-xl border-2 border-[var(--color-border)] bg-white focus:border-[var(--color-primary)] focus:outline-none transition-colors"
              />
            </div>
          </div>

          {/* Category Filters */}
          <div className="flex gap-2 mb-8 overflow-x-auto pb-2">
            {categoryOptions.map((category) => (
              <button
                key={category.id}
                onClick={() => setSelectedCategory(category.id)}
                className={`px-4 py-2 rounded-full whitespace-nowrap transition-all ${
                  selectedCategory === category.id
                    ? 'bg-[var(--color-primary)] text-white'
                    : 'bg-[var(--color-bg-accent)] text-[var(--color-text-primary)] hover:bg-[var(--color-primary)] hover:text-white'
                }`}
              >
                {category.name}
              </button>
            ))}
          </div>

          {/* Selected Places */}
          {selectedPlaces.length > 0 && (
            <div className="mb-8">
              <h3 className="mb-4">Selezionati ({selectedPlaces.length})</h3>
              <div className="flex flex-wrap gap-2">
                {selectedPlaces.map((place) => (
                  <div
                    key={place.id}
                    className="flex items-center gap-2 bg-[var(--color-bg-accent)] px-4 py-2 rounded-full"
                  >
                    <span>{place.name}</span>
                    <button onClick={() => removePlace(place.id)} className="hover:text-[var(--color-error)]">
                      <X size={16} />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Places List */}
          <div className="space-y-3 mb-8">
            {loading ? (
              <div className="flex items-center justify-center py-8 text-[var(--color-text-secondary)]">
                <Loader2 className="mr-2 animate-spin" size={18} /> Ricerca dei luoghi...
              </div>
            ) : places.length === 0 ? (
              <div className="py-6 text-center text-[var(--color-text-secondary)]">Nessun luogo trovato</div>
            ) : (
              places.map((place) => {
                const isSelected = selectedPlaces.find((p) => p.id === place.id);
                const imageSrc = place.image || apiConfig.fallback.placeholderImage;
                return (
                  <button
                    key={place.id}
                    onClick={() => togglePlace(place)}
                    className={`w-full text-left p-4 rounded-xl border-2 transition-all ${
                      isSelected
                        ? 'border-[var(--color-primary)] bg-[var(--color-bg-accent)]'
                        : 'border-[var(--color-border)] bg-white hover:border-[var(--color-primary)]'
                    }`}
                  >
                    <div className="flex items-center gap-4">
                      <div className="w-24 h-24 rounded-xl overflow-hidden flex-shrink-0 bg-[var(--color-bg-accent)]">
                        <img
                          src={imageSrc}
                          alt={place.name}
                          className="w-full h-full object-cover"
                          loading="lazy"
                        />
                      </div>
                      <div className="flex-1 min-w-0 space-y-2">
                        <div className="flex items-start justify-between gap-4">
                          <div className="min-w-0">
                            <h3 className="mb-1 truncate">{place.name}</h3>
                            <p className="text-sm text-[var(--color-text-secondary)] truncate">{place.address}</p>
                          </div>
                          {place.category && (
                            <span className="inline-flex px-3 py-1 bg-white rounded-full text-sm whitespace-nowrap">
                              {place.category}
                            </span>
                          )}
                        </div>
                        {place.description && (
                          <p className="text-sm text-[var(--color-text-secondary)] line-clamp-2">{place.description}</p>
                        )}
                      </div>
                    </div>
                  </button>
                );
              })
            )}
          </div>

          {/* Continue Button */}
          <button
            onClick={handleContinue}
            disabled={selectedPlaces.length === 0}
            className={`w-full py-3 px-6 rounded-xl transition-colors text-white ${
              selectedPlaces.length === 0
                ? 'bg-[var(--color-border)] cursor-not-allowed'
                : 'bg-[var(--color-primary)] hover:bg-[var(--color-primary-hover)]'
            }`}
          >
            Continua
          </button>
          {selectedPlaces.length === 0 && (
            <p className="mt-2 text-sm text-[var(--color-error)]">Seleziona almeno un luogo per continuare.</p>
          )}
        </div>
      </div>
    </div>
  );
}
