"use client";
import { useRef, useState } from "react";
import type { Environment } from "@/data/environments";

type Msg = { role: "user" | "assistant"; content: string };

const STARTERS = [
  "Which experiments matter most for a Moon base at 30% oxygen?",
  "Why might a fire on the Moon base be harder to put out than on the ISS?",
  "What did NASA learn about smoke detectors in microgravity?",
  "What are cool flames and why are they a safety risk?",
];

export default function Chat({ env }: { env: Environment }) {
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);

  async function send(text: string) {
    if (!text.trim() || busy) return;
    const next: Msg[] = [...msgs, { role: "user", content: text }];
    setMsgs(next);
    setInput("");
    setBusy(true);
    try {
      const r = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: next, environment: env }),
      });
      const j = await r.json();
      setMsgs([...next, { role: "assistant", content: j.reply ?? j.error ?? "Something went wrong." }]);
    } catch {
      setMsgs([...next, { role: "assistant", content: "Could not reach the chat API." }]);
    } finally {
      setBusy(false);
      setTimeout(() => endRef.current?.scrollIntoView({ behavior: "smooth" }), 50);
    }
  }

  return (
    <div className="flex h-[600px] flex-col rounded-xl border border-line bg-surface">
      <div className="flex-1 space-y-4 overflow-y-auto p-5">
        {msgs.length === 0 && (
          <div>
            <p className="text-sm text-ink-2">
              Ask about the NASA combustion investigations. Answers are grounded in the investigation data and cite the experiment they come from. Current context: <span className="text-ink">{env.name}</span>.
            </p>
            <div className="mt-4 grid gap-2 sm:grid-cols-2">
              {STARTERS.map((s) => (
                <button key={s} onClick={() => send(s)} className="rounded-lg border border-line bg-surface-2 p-3 text-left text-sm text-ink-2 hover:border-ink-3 hover:text-ink">
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}
        {msgs.map((m, i) => (
          <div key={i} className={m.role === "user" ? "ml-auto max-w-[80%] rounded-xl bg-surface-2 px-4 py-2 text-sm text-ink" : "max-w-[90%] whitespace-pre-wrap text-sm leading-relaxed text-ink-2"}>
            {m.content}
          </div>
        ))}
        {busy && <div className="text-sm text-ink-3">Thinking…</div>}
        <div ref={endRef} />
      </div>
      <form
        className="flex gap-2 border-t border-line p-3"
        onSubmit={(e) => {
          e.preventDefault();
          send(input);
        }}
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          maxLength={1000}
          placeholder="Ask about fire behaviour, materials, detection…"
          className="flex-1 rounded-lg border border-line bg-surface-2 px-3 py-2 text-sm text-ink placeholder:text-ink-3"
        />
        <button disabled={busy} className="rounded-lg bg-fire px-4 py-2 text-sm font-medium text-white disabled:opacity-50">
          Ask
        </button>
      </form>
    </div>
  );
}
