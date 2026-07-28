"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import { ArrowUp, ExternalLink, RotateCcw, Square } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";
import type { ChatCitation, DiagnosisContext } from "@/lib/types";

import { AssistantMarkdown } from "./markdown";
import { suggestedQuestions } from "./suggestions";
import type { DiagnosisUIMessage } from "./types";

/**
 * The management assistant, mounted by `components/scan/chat-sheet.tsx` inside a
 * full-height bottom sheet. Built phone-first: this is used one-handed, in the
 * sun, standing in a row of corn.
 *
 * Keep the props signature stable — the scan page mounts it.
 */
export interface DiagnosisChatProps {
  context: DiagnosisContext;
}

function textOf(message: DiagnosisUIMessage): string {
  return message.parts
    .filter((part): part is { type: "text"; text: string } => part.type === "text")
    .map((part) => part.text)
    .join("");
}

function SourceList({ citations }: { citations: ChatCitation[] }) {
  if (citations.length === 0) return null;

  return (
    <div className="mt-3 border-t pt-2.5">
      <p className="text-muted-foreground mb-1.5 text-[0.6875rem] font-semibold tracking-wide uppercase">
        Sources
      </p>
      <ol className="space-y-1.5">
        {citations.map((citation, i) => (
          <li key={`${citation.url ?? citation.title}-${i}`} className="flex gap-2 text-[0.8125rem]">
            <span className="text-muted-foreground shrink-0 tabular-nums">{i + 1}.</span>
            {citation.url ? (
              <a
                href={citation.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-primary hover:underline active:underline inline-flex min-h-9 items-start gap-1 py-1"
              >
                <span className="leading-snug">{citation.title}</span>
                <ExternalLink className="mt-0.5 size-3 shrink-0 opacity-70" aria-hidden />
              </a>
            ) : (
              <span className="text-muted-foreground leading-snug">{citation.title}</span>
            )}
          </li>
        ))}
      </ol>
    </div>
  );
}

export function DiagnosisChat({ context }: DiagnosisChatProps) {
  const [input, setInput] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  const transport = useMemo(() => new DefaultChatTransport({ api: "/api/chat" }), []);
  const { messages, sendMessage, status, error, stop, regenerate } = useChat<DiagnosisUIMessage>({
    transport,
  });

  const busy = status === "submitted" || status === "streaming";

  // Stick to the bottom while tokens arrive, unless the grower has scrolled up
  // to re-read something.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 160;
    if (nearBottom || status === "submitted") {
      bottomRef.current?.scrollIntoView({ block: "end", behavior: "smooth" });
    }
  }, [messages, status]);

  // The scan context rides along on every request rather than being captured
  // once at mount, so a re-scan behind an open sheet cannot leave the model
  // answering about the previous leaf.
  const submit = useCallback(
    (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || busy) return;
      sendMessage({ text: trimmed }, { body: { context } });
      setInput("");
    },
    [busy, context, sendMessage],
  );

  const suggestions = suggestedQuestions(context);
  const empty = messages.length === 0;

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto overscroll-contain px-4 py-4">
        {empty ? (
          <div className="flex h-full flex-col justify-end gap-3">
            <p className="text-muted-foreground text-[0.9375rem] leading-relaxed text-balance">
              Ask about managing {context.disease_label.toLowerCase()} on this field. Answers cite
              university extension and Crop Protection Network publications.
            </p>
            <div className="flex flex-col gap-2">
              {suggestions.map((question) => (
                <button
                  key={question}
                  type="button"
                  onClick={() => submit(question)}
                  className="bg-card hover:bg-accent active:bg-accent min-h-11 rounded-xl border px-3.5 py-2.5 text-left text-[0.9375rem] leading-snug transition-colors"
                >
                  {question}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="space-y-5">
            {messages.map((message) => {
              const text = textOf(message);
              const citations = message.metadata?.citations ?? [];

              if (message.role === "user") {
                return (
                  <div key={message.id} className="flex justify-end">
                    <div className="bg-primary text-primary-foreground max-w-[85%] rounded-2xl rounded-br-md px-3.5 py-2.5 text-[0.9375rem] leading-relaxed whitespace-pre-wrap">
                      {text}
                    </div>
                  </div>
                );
              }

              return (
                <div key={message.id}>
                  {text ? (
                    <AssistantMarkdown text={text} citations={citations} />
                  ) : (
                    <ThinkingDots />
                  )}
                  {text && status !== "streaming" ? <SourceList citations={citations} /> : null}
                </div>
              );
            })}

            {status === "submitted" ? <ThinkingDots /> : null}

            {error ? (
              <div className="border-destructive/40 bg-destructive/8 rounded-xl border px-3.5 py-3">
                <p className="text-[0.875rem] leading-snug">
                  {error.message || "Something went wrong reaching the assistant."}
                </p>
                <Button
                  variant="outline"
                  size="sm"
                  className="mt-2.5"
                  onClick={() => regenerate({ body: { context } })}
                >
                  <RotateCcw className="size-3.5" aria-hidden />
                  Try again
                </Button>
              </div>
            ) : null}

            <div ref={bottomRef} />
          </div>
        )}
      </div>

      <form
        className="bg-background shrink-0 border-t px-3 py-3"
        onSubmit={(event) => {
          event.preventDefault();
          submit(input);
        }}
      >
        <div className="flex items-end gap-2">
          <Textarea
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              // Enter sends on a physical keyboard; on a phone the soft keyboard
              // sends `Enter` with no modifier only when the user taps Go, and
              // the send button is always there anyway.
              if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
                event.preventDefault();
                submit(input);
              }
            }}
            rows={1}
            placeholder={`Ask about ${context.disease_label.toLowerCase()}…`}
            aria-label="Message the management assistant"
            className="max-h-32 min-h-11 flex-1 resize-none rounded-xl py-2.5 text-base"
          />
          {busy ? (
            <Button
              type="button"
              size="icon"
              variant="secondary"
              onClick={() => stop()}
              aria-label="Stop generating"
              className="size-11 shrink-0 rounded-xl"
            >
              <Square className="size-4 fill-current" aria-hidden />
            </Button>
          ) : (
            <Button
              type="submit"
              size="icon"
              disabled={!input.trim()}
              aria-label="Send"
              className={cn("size-11 shrink-0 rounded-xl")}
            >
              <ArrowUp className="size-5" aria-hidden />
            </Button>
          )}
        </div>
      </form>
    </div>
  );
}

function ThinkingDots() {
  return (
    <div className="flex items-center gap-1 py-1" role="status" aria-label="Thinking">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="bg-muted-foreground/50 size-1.5 animate-pulse rounded-full"
          style={{ animationDelay: `${i * 160}ms` }}
        />
      ))}
    </div>
  );
}

export default DiagnosisChat;
