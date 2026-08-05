import { openai } from "@ai-sdk/openai";
import {
  convertToModelMessages,
  createUIMessageStreamResponse,
  streamText,
  toUIMessageStream,
} from "ai";

import { isCorpusReady, retrieve, type RetrievedChunk } from "@/lib/rag";
import type { ChatCitation, DiagnosisContext } from "@/lib/types";
import type { DiagnosisUIMessage } from "@/components/chat/types";

/**
 * The DeGLS management assistant.
 *
 * Replaces the hosted Botpress widget the legacy Flask app embedded. Same job —
 * answer "what do I do about this?" after a diagnosis — but self-hosted, with
 * retrieval over the extension corpus in `corpus/` so answers cite a real
 * publication instead of a model's recollection of one.
 *
 * OPENAI_API_KEY is read here, server-side, and never sent to the client. It is
 * deliberately NOT a NEXT_PUBLIC_ variable: anything with that prefix is inlined
 * into the browser bundle, which is how the previous version of this project
 * leaked a key into a public repository.
 */

export const runtime = "nodejs";
export const maxDuration = 60;

// gpt-5.4-nano: $0.20/1M in, $1.25/1M out - roughly 3.7x cheaper than 5.4-mini
// ($0.75/$4.50). This route synthesises an answer from passages RAG has already
// selected rather than reasoning open-endedly, which is the workload small models
// handle well, so the cheaper tier is the right default here.
//
// If citation formatting or answer quality degrades, this is a one-variable
// rollback: set OPENAI_CHAT_MODEL=gpt-5.4-mini in the Vercel project settings.
// No redeploy of code required.
const CHAT_MODEL = process.env.OPENAI_CHAT_MODEL ?? "gpt-5.4-nano";

// Request ceilings for the public, unauthenticated chat endpoint. See the cost
// guard in POST below for why these exist.
const MAX_MESSAGES = 40;
const MAX_MESSAGE_CHARS = 4_000;
const MAX_TOTAL_CHARS = 24_000;

const RETRIEVAL_K = 6;

interface ChatRequestBody {
  messages: DiagnosisUIMessage[];
  context?: DiagnosisContext;
}

/** Flatten a UI message's text parts — what we embed as the retrieval query. */
function messageText(message: DiagnosisUIMessage | undefined): string {
  if (!message) return "";
  return message.parts
    .filter((part): part is { type: "text"; text: string } => part.type === "text")
    .map((part) => part.text)
    .join(" ")
    .trim();
}

function describeContext(context: DiagnosisContext | undefined): string {
  if (!context) {
    return "No diagnosis is attached to this conversation. Ask the grower for the disease, severity, and growth stage before giving specific advice.";
  }

  const lines = context.unclassified
    ? [
        "- The scan did not produce a usable classification. Do NOT name a disease as",
        "  identified and do NOT quote a severity figure. Answer generally, and ask the",
        "  grower what they are seeing before giving specific advice.",
      ]
    : [
        `- Disease identified: ${context.disease_label} (model class \`${context.disease_code}\`)`,
        `- Model confidence: ${(context.confidence * 100).toFixed(0)}%`,
        `- Lesion severity: ${context.severity_percent.toFixed(1)}% of the analyzed leaf area`,
      ];
  if (context.corn_hybrid) lines.push(`- Corn hybrid: ${context.corn_hybrid}`);
  if (context.location) lines.push(`- Location: ${context.location}`);
  if (context.date) lines.push(`- Scan date: ${context.date}`);
  return lines.join("\n");
}

interface GroupedSource {
  title: string;
  publisher: string;
  date?: string;
  url?: string;
  passages: string[];
}

/**
 * Collapse retrieved chunks to one entry per source document.
 *
 * Retrieval routinely returns three chunks of the same Purdue bulletin. If those
 * were numbered [1][2][3] in the prompt while the UI showed a URL-deduplicated
 * list, the model's [3] would point at an entirely different publication in the
 * rendered citations. Numbering by document keeps the prompt and the UI in
 * lockstep by construction.
 */
function groupSources(chunks: RetrievedChunk[]): GroupedSource[] {
  const byKey = new Map<string, GroupedSource>();
  for (const chunk of chunks) {
    const key = chunk.source.url ?? chunk.source.title;
    const existing = byKey.get(key);
    if (existing) {
      existing.passages.push(chunk.text);
      continue;
    }
    byKey.set(key, {
      title: chunk.source.title,
      publisher: chunk.source.publisher,
      date: chunk.source.date,
      url: chunk.source.url,
      passages: [chunk.text],
    });
  }
  return [...byKey.values()];
}

function renderSources(sources: GroupedSource[]): string {
  return sources
    .map((source, i) => {
      const meta = [source.publisher, source.date].filter(Boolean).join(", ");
      const url = source.url ? `\nURL: ${source.url}` : "";
      return `[${i + 1}] ${source.title} — ${meta}${url}\n\n${source.passages.join("\n\n…\n\n")}`;
    })
    .join("\n\n---\n\n");
}

function toCitations(sources: GroupedSource[]): ChatCitation[] {
  return sources.map((source) => ({ title: source.title, url: source.url }));
}

function buildSystemPrompt(
  context: DiagnosisContext | undefined,
  sources: GroupedSource[],
): string {
  const grounded = sources.length > 0;

  const base = `You are the DeGLS management assistant. A grower has just photographed a corn leaf; a computer-vision model classified the disease and measured lesion severity, and now they want to know what to do about it. They are often standing in the field on a phone.

Answer directly and commit to a recommendation. Lead with the answer, then the reasoning. Short paragraphs and tight bullet lists; no preamble, no restating the question.

## This scan

${describeContext(context)}

Note on severity: the percentage is lesion pixels as a share of the analyzed leaf's area from a single photograph. It is a measurement of one leaf, not a whole-field rating and not a calibrated economic threshold. Extension thresholds are usually expressed in terms of where lesions sit in the canopy relative to the ear leaf, and what fraction of plants across the field show them — so when severity matters to the answer, tell the grower what to go check in the field rather than treating this number as the decision.`;

  const citationRules = `## Sources and citations

${
  grounded
    ? `You have been given passages from published university extension and Crop Protection Network material below. Ground your answer in them.

- Cite with a bracketed number matching the source list, e.g. "apply at VT/R1 [2]". Put the marker at the end of the sentence it supports.
- Cite every specific claim you take from the passages: numbers, temperature and humidity ranges, growth stages, thresholds, yield-loss figures, product efficacy ratings.
- Do not invent citation numbers. Only [1] through [${sources.length}] exist.
- If the passages do not cover part of the question, answer that part from general agronomic knowledge and say plainly that it is not from the cited sources.`
    : `No corpus passages were retrieved for this question — either the index has not been built or nothing in it was relevant.

- Say so in one short line at the top, e.g. "Nothing in my sources covers this directly, so this is general guidance."
- Then answer from general agronomic knowledge.
- Do not cite anything. Do not invent a publication, a bulletin number, or a URL.`
}

Never fabricate details you cannot source: specific fungicide application rates, product names, spray intervals, pre-harvest or re-entry intervals, tank-mix partners, or anything about legal/label requirements. If you were not given it, say what the grower should look up — the product label, the current CPN fungicide efficacy table — instead of guessing at a number. A wrong rate is worse than no rate.`;

  const passages = grounded ? `\n\n## Retrieved passages\n\n${renderSources(sources)}` : "";

  return `${base}\n\n${citationRules}${passages}`;
}

export async function POST(req: Request) {
  if (!process.env.OPENAI_API_KEY) {
    return Response.json(
      {
        error:
          "The assistant is not configured: OPENAI_API_KEY is missing on the server. Set it in .env.local (never NEXT_PUBLIC_) and restart.",
      },
      { status: 503 },
    );
  }

  let body: ChatRequestBody;
  try {
    body = (await req.json()) as ChatRequestBody;
  } catch {
    return Response.json({ error: "Malformed request body." }, { status: 400 });
  }

  const messages = Array.isArray(body.messages) ? body.messages : [];
  if (messages.length === 0) {
    return Response.json({ error: "No messages supplied." }, { status: 400 });
  }

  // Cost guard. This route is public and unauthenticated, and the model has a
  // 400k-token context window, so an unbounded body is a direct route to
  // draining the account's OpenAI credit. Cap the conversation before it ever
  // reaches the provider. These ceilings are far above any genuine field
  // question - the longest suggested prompt is ~90 characters.
  if (messages.length > MAX_MESSAGES) {
    return Response.json(
      { error: `Conversation too long: ${messages.length} messages, limit ${MAX_MESSAGES}.` },
      { status: 413 },
    );
  }

  const totalChars = messages.reduce((sum, message) => sum + messageText(message).length, 0);
  if (totalChars > MAX_TOTAL_CHARS) {
    return Response.json(
      { error: `Conversation too large: ${totalChars} characters, limit ${MAX_TOTAL_CHARS}.` },
      { status: 413 },
    );
  }

  if (messages.some((message) => messageText(message).length > MAX_MESSAGE_CHARS)) {
    return Response.json(
      { error: `Message too long; limit ${MAX_MESSAGE_CHARS} characters.` },
      { status: 413 },
    );
  }

  const context = body.context;
  const question = messageText(messages[messages.length - 1]);

  // Bias retrieval toward the diagnosed disease. "Should I spray?" on its own
  // matches fungicide-timing chunks for every disease in the corpus; prefixing
  // the label pulls the right ones to the top.
  const retrievalQuery =
    context && !context.unclassified ? `${context.disease_label} in corn. ${question}` : question;

  let chunks: RetrievedChunk[] = [];
  if (isCorpusReady()) {
    chunks = await retrieve(retrievalQuery, RETRIEVAL_K);
  }

  const sources = groupSources(chunks);
  const citations = toCitations(sources);
  const grounded = sources.length > 0;

  const result = streamText({
    model: openai(CHAT_MODEL),
    system: buildSystemPrompt(context, sources),
    messages: await convertToModelMessages(messages),
    providerOptions: {
      // Keep time-to-first-token low; this is a field tool, not a research task.
      openai: { reasoningEffort: "low" },
    },
  });

  return createUIMessageStreamResponse({
    stream: toUIMessageStream({
      stream: result.stream,
      originalMessages: messages,
      messageMetadata: ({ part }) =>
        part.type === "start" ? { citations, grounded } : undefined,
    }),
  });
}
