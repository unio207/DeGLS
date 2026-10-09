import { Fragment, type ReactNode } from "react";

import type { ChatCitation } from "@/lib/types";

/**
 * A deliberately small markdown renderer for assistant replies.
 *
 * Pulling in a full markdown pipeline for what the model actually emits here —
 * paragraphs, bullets, the odd bold run, and bracketed citation markers — would
 * cost more bundle than the whole chat UI. This covers those cases and, more
 * importantly, turns `[2]` into a tappable link to source 2, which a generic
 * renderer would not do.
 */

const INLINE_PATTERN = /(\*\*[^*]+\*\*)|(\[\d+(?:\s*,\s*\d+)*\])/g;

function CitationChip({ index, citation }: { index: number; citation?: ChatCitation }) {
  const label = <span className="tabular-nums">{index}</span>;
  // `after:-inset-2` grows the tap target to ~32px without changing how the
  // marker looks — these are thumb-sized targets in a field, not mouse targets.
  const className =
    "relative inline-flex h-4 min-w-4 items-center justify-center rounded-[4px] px-1 align-super text-[0.625rem] font-semibold leading-none after:absolute after:-inset-2 after:content-['']";

  if (!citation?.url) {
    return (
      <span
        className={`${className} bg-muted text-muted-foreground`}
        title={citation?.title ?? `Source ${index}`}
      >
        {label}
      </span>
    );
  }

  return (
    <a
      href={citation.url}
      target="_blank"
      rel="noopener noreferrer"
      title={citation.title}
      className={`${className} bg-primary/12 text-primary hover:bg-primary/20 active:bg-primary/25`}
    >
      {label}
    </a>
  );
}

function renderInline(text: string, citations: ChatCitation[], keyPrefix: string): ReactNode[] {
  const nodes: ReactNode[] = [];
  let last = 0;
  let match: RegExpExecArray | null;
  const pattern = new RegExp(INLINE_PATTERN.source, "g");

  while ((match = pattern.exec(text)) !== null) {
    if (match.index > last) nodes.push(text.slice(last, match.index));

    if (match[1]) {
      nodes.push(
        <strong key={`${keyPrefix}-b-${match.index}`} className="font-semibold">
          {match[1].slice(2, -2)}
        </strong>,
      );
    } else if (match[2]) {
      const numbers = match[2]
        .slice(1, -1)
        .split(",")
        .map((n) => Number.parseInt(n.trim(), 10))
        .filter((n) => Number.isFinite(n) && n >= 1 && n <= citations.length);

      if (numbers.length === 0) {
        // A marker the model made up, or one pointing past the source list.
        // Drop it rather than render a dead reference.
        nodes.push("");
      } else {
        nodes.push(
          <Fragment key={`${keyPrefix}-c-${match.index}`}>
            {numbers.map((n) => (
              <CitationChip key={n} index={n} citation={citations[n - 1]} />
            ))}
          </Fragment>,
        );
      }
    }
    last = match.index + match[0].length;
  }

  if (last < text.length) nodes.push(text.slice(last));
  return nodes;
}

interface Block {
  type: "p" | "ul" | "ol" | "h";
  lines: string[];
}

function parseBlocks(text: string): Block[] {
  const blocks: Block[] = [];
  // Index of the block a following line may continue. A blank line or a heading
  // resets it to -1, so nothing bridges a paragraph break.
  let openIndex = -1;

  for (const rawLine of text.split("\n")) {
    const line = rawLine.trimEnd();

    if (!line.trim()) {
      openIndex = -1;
      continue;
    }

    const heading = line.match(/^\s*#{1,6}\s+(.*)$/);
    if (heading) {
      blocks.push({ type: "h", lines: [heading[1]] });
      openIndex = -1;
      continue;
    }

    const bullet = line.match(/^\s*[-*+]\s+(.*)$/);
    const numbered = bullet ? null : line.match(/^\s*\d+[.)]\s+(.*)$/);
    const type: Block["type"] = bullet ? "ul" : numbered ? "ol" : "p";
    const content = bullet?.[1] ?? numbered?.[1] ?? line;

    const open = openIndex >= 0 ? blocks[openIndex] : undefined;
    if (open && open.type === type) {
      open.lines.push(content);
      continue;
    }

    blocks.push({ type, lines: [content] });
    openIndex = blocks.length - 1;
  }

  return blocks;
}

export function AssistantMarkdown({
  text,
  citations = [],
}: {
  text: string;
  citations?: ChatCitation[];
}) {
  const blocks = parseBlocks(text);

  return (
    <div className="space-y-2.5 text-[0.9375rem] leading-relaxed">
      {blocks.map((block, i) => {
        if (block.type === "h") {
          return (
            <p key={i} className="pt-1 text-[0.8125rem] font-semibold tracking-wide uppercase">
              {renderInline(block.lines[0], citations, `h${i}`)}
            </p>
          );
        }
        if (block.type === "ul" || block.type === "ol") {
          const List = block.type === "ul" ? "ul" : "ol";
          return (
            <List
              key={i}
              className={
                block.type === "ul"
                  ? "list-disc space-y-1.5 pl-5 marker:text-muted-foreground"
                  : "list-decimal space-y-1.5 pl-5 marker:text-muted-foreground"
              }
            >
              {block.lines.map((line, j) => (
                <li key={j}>{renderInline(line, citations, `l${i}-${j}`)}</li>
              ))}
            </List>
          );
        }
        return <p key={i}>{renderInline(block.lines.join(" "), citations, `p${i}`)}</p>;
      })}
    </div>
  );
}
