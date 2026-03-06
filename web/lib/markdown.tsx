import React from 'react';

/**
 * Renders text with markdown bold syntax (**text**) as actual bold text.
 * @param text - The text containing markdown bold syntax
 * @returns React elements with bold formatting
 */
export function renderMarkdownBold(text: string): React.ReactNode {
  if (!text) return null;

  // Split by **bold** pattern
  const parts = text.split(/(\*\*.*?\*\*)/g);

  return parts.map((part, index) => {
    // Check if this part is bold (surrounded by **)
    if (part.startsWith('**') && part.endsWith('**')) {
      // Remove ** and render as bold
      const boldText = part.slice(2, -2);
      return <strong key={index}>{boldText}</strong>;
    }
    // Regular text
    return <React.Fragment key={index}>{part}</React.Fragment>;
  });
}
