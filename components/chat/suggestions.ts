import type { DiagnosisContext, DiseaseCode } from "@/lib/types";

/**
 * Prompts for the chat, seeded from the scan.
 *
 * A blank chat box in a field is a dead end — the grower knows what they want
 * but not what this thing can answer. `suggestedQuestions` opens the thread;
 * `followUpQuestions` keeps a row of chips under every reply, which is what
 * testers asked for ("lined up like how it is at the start").
 *
 * Both are pure functions of the scan and what has already been asked. No
 * second model call: a suggestion that takes a network round-trip to appear is
 * worse than no suggestion, and this has to be instant on a phone.
 */

const BY_DISEASE: Record<DiseaseCode, string[]> = {
  corn_gls: [
    "Is this worth spraying?",
    "Is it gray leaf spot or bacterial leaf streak?",
    "What weather is driving it?",
    "What do I change next season?",
  ],
  corn_nlb: [
    "Is this worth spraying?",
    "When is the yield damage done?",
    "Is it blight or Goss's wilt?",
    "What resistance should I ask for?",
  ],
  corn_rust: [
    "Does rust justify a fungicide?",
    "Common rust or southern rust?",
    "Where did this come from?",
    "What do I watch for next?",
  ],
};

/** Used when the scan produced no classification to key the questions off. */
const UNCLASSIFIED: string[] = [
  "What's going around in corn right now?",
  "Gray leaf spot or northern leaf blight?",
  "What do I check before spraying?",
  "How do I take a better leaf photo?",
];

/**
 * Follow-ups that read sensibly after any answer, ordered roughly the way a
 * conversation in the field actually goes: decide, then time it, then cost it,
 * then plan the season.
 */
const FOLLOW_UPS: string[] = [
  "How urgent is this?",
  "When should I spray?",
  "What could this cost me in yield?",
  "How fast does it spread?",
  "What should I scout next?",
  "Will it hit my other fields?",
  "What do I change next season?",
  "How do I confirm it's this?",
];

function normalize(text: string): string {
  return text.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
}

/** Opening chips, shown on the empty state. */
export function suggestedQuestions(context: DiagnosisContext): string[] {
  if (context.unclassified) {
    const generic = [...UNCLASSIFIED];
    if (context.corn_hybrid) generic.push(`What should I watch for on ${context.corn_hybrid}?`);
    return generic.slice(0, 4);
  }

  const questions = [...(BY_DISEASE[context.disease_code] ?? BY_DISEASE.corn_gls)];

  if (context.severity_percent >= 5) {
    questions[0] = `${context.severity_percent.toFixed(0)}% on this leaf — is that bad?`;
  }
  if (context.corn_hybrid) {
    questions.push(`What does this mean for ${context.corn_hybrid}?`);
  }

  return questions.slice(0, 4);
}

/**
 * Chips shown under the latest reply. `asked` is the text of every user turn so
 * far; anything already asked is dropped so the same chip never reappears.
 */
export function followUpQuestions(context: DiagnosisContext, asked: string[]): string[] {
  const used = new Set(asked.map(normalize));

  const pool = [
    // Anything from the opening set the grower skipped is still the strongest
    // suggestion — it was chosen for this disease.
    ...suggestedQuestions(context),
    ...FOLLOW_UPS,
  ];

  const out: string[] = [];
  for (const question of pool) {
    const key = normalize(question);
    if (used.has(key)) continue;
    used.add(key);
    out.push(question);
    if (out.length === 3) break;
  }
  return out;
}
