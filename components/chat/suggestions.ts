import type { DiagnosisContext, DiseaseCode } from "@/lib/types";

/**
 * Opening questions, seeded from the scan.
 *
 * A blank chat box in a field is a dead end — the grower knows what they want
 * but not what this thing can answer. These are the four questions the corpus
 * actually covers well for each disease, phrased the way someone standing in
 * the row would ask them.
 */

const BY_DISEASE: Record<DiseaseCode, string[]> = {
  corn_gls: [
    "Is this worth spraying?",
    "How do I tell this apart from bacterial leaf streak?",
    "What weather is driving gray leaf spot right now?",
    "What should I change next season?",
  ],
  corn_nlb: [
    "Is this worth spraying?",
    "When is the yield damage actually done?",
    "How do I confirm it's northern leaf blight and not Goss's wilt?",
    "What resistance should I ask my seed dealer about?",
  ],
  corn_rust: [
    "Does common rust ever justify a fungicide?",
    "How do I tell common rust from southern rust?",
    "Where did this come from mid-season?",
    "What should I watch for from here?",
  ],
};

/** Used when the scan produced no classification to key the questions off. */
const UNCLASSIFIED: string[] = [
  "What are the common corn leaf diseases right now?",
  "How do I tell gray leaf spot from northern leaf blight?",
  "What should I look for before deciding to spray?",
  "How should I photograph a leaf for a better reading?",
];

export function suggestedQuestions(context: DiagnosisContext): string[] {
  if (context.unclassified) {
    const generic = [...UNCLASSIFIED];
    if (context.corn_hybrid) generic.push(`What should I watch for on ${context.corn_hybrid}?`);
    return generic.slice(0, 4);
  }

  const base = BY_DISEASE[context.disease_code] ?? BY_DISEASE.corn_gls;
  const questions = [...base];

  if (context.severity_percent >= 5) {
    questions[0] = `${context.severity_percent.toFixed(0)}% on this leaf — is that bad?`;
  }
  if (context.corn_hybrid) {
    questions.push(`What does this mean for ${context.corn_hybrid}?`);
  }

  return questions.slice(0, 4);
}
