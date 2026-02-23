// Gestione dell'autenticazione con JWT

const AUTH_TOKEN_KEY = 'auth_token';
const BACKEND_BASE = `${import.meta.env.BASE_URL}backend`;

export interface AuthResponse {
  token: string;
  isNewUser: boolean;
  hasProfile: boolean;
  profile: Record<string, number> | null;
  favoritePlaces: any[];
}

export interface VerifyResponse {
  valid: boolean;
  hasProfile: boolean;
}

// Ottiene il token salvato
export const getToken = (): string | null => {
  return localStorage.getItem(AUTH_TOKEN_KEY);
};

// Salva il token
export const setToken = (token: string): void => {
  localStorage.setItem(AUTH_TOKEN_KEY, token);
};

// Rimuove il token
export const clearToken = (): void => {
  localStorage.removeItem(AUTH_TOKEN_KEY);
};

// Verifica se l'utente è autenticato
export const isAuthenticated = (): boolean => {
  return !!getToken();
};

// Headers comuni per le richieste autenticate
export const getAuthHeaders = (): HeadersInit => {
  const token = getToken();
  return {
    'Content-Type': 'application/json',
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };
};

// Login con nickname
export const login = async (nickname: string): Promise<AuthResponse> => {
  const response = await fetch(`${BACKEND_BASE}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ nickname }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ error: 'Errore di login' }));
    throw new Error(error.error || 'Errore di login');
  }

  const data = await response.json();
  setToken(data.token);
  return data;
};

// Verifica il token
export const verifyToken = async (): Promise<VerifyResponse> => {
  const token = getToken();
  if (!token) {
    return { valid: false, hasProfile: false };
  }

  try {
    const response = await fetch(`${BACKEND_BASE}/auth/verify`, {
      method: 'POST',
      headers: getAuthHeaders(),
    });

    if (!response.ok) {
      clearToken();
      return { valid: false, hasProfile: false };
    }

    return await response.json();
  } catch {
    return { valid: false, hasProfile: false };
  }
};

// Logout
export const logout = async (): Promise<void> => {
  try {
    await fetch(`${BACKEND_BASE}/auth/logout`, {
      method: 'POST',
      headers: getAuthHeaders(),
    });
  } finally {
    clearToken();
  }
};
