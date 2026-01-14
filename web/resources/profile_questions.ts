interface ProfileQuestion {
  id: string;
  text: string;
  min_label: string;
  max_label: string;
  type: "number" | "boolean";
}

export const questions: ProfileQuestion[] = [
  {
    id: "age",
    text: "Quanti anni hai?",
    min_label: "0",
    max_label: "100",
    type: "number"
  },
  {
    id: "asd",
    text: "Sei stato diagnosticato con disturbo dello spettro autistico?",
    min_label: "No",
    max_label: "Sì",
    type: "boolean"
  }
  
];