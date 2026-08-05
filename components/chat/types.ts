/**
 * Shared between `app/api/chat/route.ts` and `components/chat/diagnosis-chat.tsx`.
 *
 * The definitions live in `lib/types.ts` alongside the rest of the contract —
 * `lib/history.ts` persists these messages and cannot import from `components/`
 * without a cycle. Re-exported here so existing import paths keep working.
 */

export type { ChatMessageMetadata, DiagnosisUIMessage } from "@/lib/types";
