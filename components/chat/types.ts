import type { UIMessage } from "ai";
import type { ChatCitation } from "@/lib/types";

/**
 * Shared between `app/api/chat/route.ts` and `components/chat/diagnosis-chat.tsx`.
 * Types only — no runtime code — so the server route can import it freely.
 */

/**
 * Per-message metadata streamed from the route.
 *
 * Citations are attached at the `start` part rather than parsed out of the
 * model's text: retrieval has already happened by then, so the UI can render
 * the source list immediately and it cannot drift from what was actually
 * retrieved. `citations` is empty when nothing cleared the relevance floor.
 */
export interface ChatMessageMetadata {
  citations?: ChatCitation[];
  /** False when the answer came from the diagnosis alone (no corpus hits). */
  grounded?: boolean;
}

export type DiagnosisUIMessage = UIMessage<ChatMessageMetadata>;
