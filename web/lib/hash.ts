export const sha256Hex = async (input: string): Promise<string> => {
  if (!globalThis.crypto?.subtle) {
    throw new Error('Web Crypto API non disponibile');
  }

  const data = new TextEncoder().encode(input);
  const digest = await globalThis.crypto.subtle.digest('SHA-256', data);
  return Array.from(new Uint8Array(digest))
    .map((byte) => byte.toString(16).padStart(2, '0'))
    .join('');
};
