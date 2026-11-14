import investigations from "@/data/investigations.json";

// Grounded Q&A over the investigations file using the Claude API.
// Needs ANTHROPIC_API_KEY (server-side env var; never exposed to the browser).

const MAX_MSG_CHARS = 1000;
const MAX_HISTORY = 10;
const LIMIT = Number(process.env.CHAT_RATE_LIMIT ?? 20); // requests per IP per window
const WINDOW_MS = 10 * 60 * 1000;

// Best-effort per-IP limiter. Memory is per server instance, so on Vercel it resets
// between cold starts — enough to stop casual abuse. Also set a monthly spend limit
// on the Anthropic console.
const hits = new Map<string, number[]>();
function limited(ip: string) {
  const now = Date.now();
  const recent = (hits.get(ip) ?? []).filter((t) => now - t < WINDOW_MS);
  recent.push(now);
  hits.set(ip, recent);
  if (hits.size > 5000) hits.clear();
  return recent.length > LIMIT;
}

const SYSTEM = `You are a fire-safety research assistant for NASA microgravity combustion data (Physical Sciences Informatics, PSI).
Answer ONLY from the investigation records below. After each claim, cite the investigation name in square brackets, e.g. [BRE].
If the records do not contain the answer, say so plainly and name the closest relevant investigation — never invent numbers.
Stay on topic (fire safety, combustion, spacecraft and habitat atmospheres). Politely decline anything unrelated.
Keep answers under 180 words, plain language, aimed at aerospace engineers and students.

INVESTIGATION RECORDS:
${JSON.stringify(investigations)}`;

type Msg = { role: "user" | "assistant"; content: string };

export async function POST(req: Request) {
  const key = process.env.ANTHROPIC_API_KEY;
  if (!key) {
    return Response.json({ reply: "The AI assistant isn't configured on this deployment yet (missing ANTHROPIC_API_KEY)." });
  }

  const ip = req.headers.get("x-forwarded-for")?.split(",")[0]?.trim() || "unknown";
  if (limited(ip)) {
    return Response.json({ reply: "You've sent a lot of questions — please wait a few minutes and try again." }, { status: 429 });
  }

  let body: { messages?: Msg[]; environment?: unknown };
  try {
    body = await req.json();
  } catch {
    return Response.json({ error: "Invalid request." }, { status: 400 });
  }

  const messages = (Array.isArray(body.messages) ? body.messages : [])
    .filter((m) => (m.role === "user" || m.role === "assistant") && typeof m.content === "string" && m.content.trim())
    .slice(-MAX_HISTORY)
    .map((m) => ({ role: m.role, content: m.content.slice(0, MAX_MSG_CHARS) }));
  while (messages.length && messages[0].role !== "user") messages.shift();
  if (!messages.length || messages[messages.length - 1].role !== "user") {
    return Response.json({ error: "Send a question." }, { status: 400 });
  }

  const env = body.environment ? `\n\nThe user is currently looking at this habitat: ${JSON.stringify(body.environment).slice(0, 500)}.` : "";

  try {
    const r = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: { "x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json" },
      body: JSON.stringify({
        model: process.env.CLAUDE_MODEL || "claude-sonnet-5-5",
        max_tokens: 700,
        system: SYSTEM + env,
        messages,
      }),
      signal: AbortSignal.timeout(60_000),
    });
    if (!r.ok) {
      console.error("Claude API error", r.status, (await r.text()).slice(0, 500));
      return Response.json({ reply: "The AI assistant is temporarily unavailable. Please try again in a moment." }, { status: 502 });
    }
    const j = await r.json();
    const reply = (j.content as { type: string; text?: string }[]).filter((b) => b.type === "text").map((b) => b.text).join("\n");
    return Response.json({ reply });
  } catch (e) {
    console.error("Chat route failed", e);
    return Response.json({ reply: "The AI assistant timed out. Please try again." }, { status: 504 });
  }
}
