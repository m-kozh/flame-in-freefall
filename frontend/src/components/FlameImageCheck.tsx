"use client";
import { useRef, useState } from "react";
import { INVESTIGATIONS } from "./Explorer";

// Measures a flame photo in the browser (no upload, no API): where the flame is,
// its shape (tall teardrop vs round vs flat front) and colour (blue vs yellow),
// then explains what that says about gravity/buoyancy and soot, and links the
// NASA investigations that studied that kind of flame.

type Kind = "microgravity" | "earth" | "spreading" | "unclear" | "none";
type Result = {
  kind: Kind;
  aspect: number;
  taper: number;
  hollow: boolean;
  blueFrac: number;
  yellowFrac: number;
  areaFrac: number;
};

const SAMPLES = [
  { src: "/samples/microgravity-flame.png", label: "Microgravity flame" },
  { src: "/samples/earth-candle.png", label: "Earth candle" },
  { src: "/samples/spreading-flame.png", label: "Flame spreading on a sample" },
];

const MATCHES: Record<Kind, string[]> = {
  microgravity: ["sflame", "flamedesign", "bre", "flex", "cfig"],
  earth: ["bre", "saffire13", "flex"],
  spreading: ["bass2", "saffire13", "saffire45"],
  unclear: [],
  none: ["flex"],
};

function analyze(img: HTMLImageElement, canvas: HTMLCanvasElement): Result {
  const scale = Math.min(1, 320 / Math.max(img.naturalWidth, img.naturalHeight));
  const W = Math.max(1, Math.round(img.naturalWidth * scale));
  const H = Math.max(1, Math.round(img.naturalHeight * scale));
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
  ctx.drawImage(img, 0, 0, W, H);
  const { data } = ctx.getImageData(0, 0, W, H);
  const N = W * H;
  const V = new Float32Array(N);
  const S = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const r = data[i * 4], g = data[i * 4 + 1], b = data[i * 4 + 2];
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
    V[i] = mx;
    S[i] = mx - mn;
  }
  // Background brightness = median of the image border; flame = clearly brighter AND coloured
  const border: number[] = [];
  const m = Math.max(1, Math.round(Math.min(W, H) * 0.04));
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) if (x < m || y < m || x >= W - m || y >= H - m) border.push(V[y * W + x]);
  border.sort((a, b) => a - b);
  const bg = border[Math.floor(border.length / 2)] ?? 0;
  const sorted = Array.from(V).sort((a, b) => a - b);
  const top = sorted[Math.floor(sorted.length * 0.995)] ?? 255;
  if (top - bg < 40) return { kind: "none", aspect: 0, taper: 0, hollow: false, blueFrac: 0, yellowFrac: 0, areaFrac: 0 };
  const thr = bg + 0.3 * (top - bg);

  const mask = new Uint8Array(N);
  const xs: number[] = [], ys: number[] = [];
  let blue = 0, yellow = 0;
  for (let i = 0; i < N; i++) {
    if (V[i] > thr && S[i] > 35) {
      mask[i] = 1;
      xs.push(i % W);
      ys.push(Math.floor(i / W));
      const r = data[i * 4], g = data[i * 4 + 1], b = data[i * 4 + 2];
      if (b > r + 20 && b >= g) blue++;
      else if (r >= b + 40) yellow++;
    }
  }
  const count = xs.length;
  if (count < N * 0.002) return { kind: "none", aspect: 0, taper: 0, hollow: false, blueFrac: 0, yellowFrac: 0, areaFrac: count / N };

  // Robust bounding box (2nd–98th percentile) so stray pixels don't stretch it
  const q = (arr: number[], p: number) => [...arr].sort((a, b) => a - b)[Math.floor((arr.length - 1) * p)];
  const x0 = q(xs, 0.02), x1 = q(xs, 0.98), y0 = q(ys, 0.02), y1 = q(ys, 0.98);
  const w = Math.max(1, x1 - x0), h = Math.max(1, y1 - y0);
  // Row extent = leftmost→rightmost flame pixel (works for filled flames and thin shells alike)
  const rowExtent = (y: number) => {
    let l = -1, r = -1;
    for (let x = x0; x <= x1; x++)
      if (mask[y * W + x]) {
        if (l < 0) l = x;
        r = x;
      }
    return l < 0 ? 0 : r - l + 1;
  };
  const avgRows = (a: number, b: number) => {
    let s = 0, n = 0;
    for (let y = Math.round(a); y <= Math.round(b); y++) {
      s += rowExtent(y);
      n++;
    }
    return n ? s / n : 0;
  };
  let maxW = 1;
  for (let y = y0; y <= y1; y++) maxW = Math.max(maxW, rowExtent(y));
  const topW = avgRows(y0, y0 + h * 0.25);
  const botW = avgRows(y1 - h * 0.25, y1);
  const taper = topW / maxW; // tip narrowness: teardrop ≈ 0.3–0.5, round ≈ 0.6–0.8
  const symmetry = botW > 0 ? topW / botW : 1; // round flames look the same top and bottom
  // Hollow centre = flame is a thin shell (typical spherical diffusion flame seen as a ring)
  let centre = 0, centreN = 0;
  for (let y = Math.round(y0 + h * 0.35); y <= Math.round(y1 - h * 0.35); y++)
    for (let x = Math.round(x0 + w * 0.35); x <= Math.round(x1 - w * 0.35); x++) {
      centre += mask[y * W + x];
      centreN++;
    }
  const hollow = centreN > 0 && centre / centreN < 0.3;
  const aspect = h / w;

  let kind: Kind = "unclear";
  if (aspect < 0.6) kind = "spreading";
  else if (aspect >= 0.7 && aspect <= 1.4 && taper >= 0.55 && symmetry >= 0.7 && symmetry <= 1.4) kind = "microgravity";
  else if (aspect > 1.5 || (aspect > 1.25 && taper < 0.55)) kind = "earth"; // buoyancy makes flames tall

  // Overlay: tint detected flame pixels and draw the bounding box
  const overlay = ctx.getImageData(0, 0, W, H);
  for (let i = 0; i < N; i++)
    if (mask[i]) {
      overlay.data[i * 4] = Math.min(255, overlay.data[i * 4] * 0.6 + 100);
      overlay.data[i * 4 + 1] = Math.min(255, overlay.data[i * 4 + 1] * 0.6 + 100);
      overlay.data[i * 4 + 2] = Math.min(255, overlay.data[i * 4 + 2] * 0.6 + 100);
    }
  ctx.putImageData(overlay, 0, 0);
  ctx.strokeStyle = "#fab219";
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 4]);
  ctx.strokeRect(x0, y0, w, h);

  return { kind, aspect, taper, hollow, blueFrac: blue / count, yellowFrac: yellow / count, areaFrac: count / N };
}

const VERDICT: Record<Kind, { title: string; body: string }> = {
  microgravity: {
    title: "Looks like a microgravity flame",
    body: "The flame is roughly round instead of a tall teardrop. Without gravity there is no buoyancy — hot gas doesn't rise — so oxygen reaches the flame only by slow diffusion from all sides. That's why flames in NASA's ISS experiments are near-spherical, and why they can survive in still air at lower oxygen and then quench differently than on Earth.",
  },
  earth: {
    title: "Looks like a normal-gravity (Earth) flame",
    body: "The flame is tall and tapered — the classic teardrop. Buoyancy pulls hot gas upward and drags fresh air in from below, feeding the flame. In orbit this convection disappears, which is exactly why ground tests can't predict how materials burn in a spacecraft.",
  },
  spreading: {
    title: "Looks like a flame front spreading along a sample",
    body: "The flame is much wider than it is tall — a flame front travelling along a thin material. This is the geometry NASA used to measure flame spread over fabrics and plastics (BASS-II on the ISS, Saffire in an uncrewed Cygnus vehicle).",
  },
  unclear: {
    title: "Shape is ambiguous",
    body: "The flame doesn't clearly match a round microgravity flame, an Earth teardrop or a spreading front. Try a tighter crop with the whole flame visible on a dark background.",
  },
  none: {
    title: "No clear flame found",
    body: "Nothing in the image is clearly brighter and more coloured than the background. Careful: some fires are invisible to normal cameras — FLEX found 'cool flames' that keep burning after the visible flame goes out. A dark image is not proof that nothing is burning.",
  },
};

export default function FlameImageCheck() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [res, setRes] = useState<Result | null>(null);
  const [name, setName] = useState<string>("");
  const [error, setError] = useState<string>("");

  function load(src: string, label: string) {
    setError("");
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      try {
        setRes(analyze(img, canvasRef.current!));
        setName(label);
      } catch {
        setError("Couldn't read that image. Try a JPG or PNG.");
      }
    };
    img.onerror = () => setError("Couldn't open that image. Try a JPG or PNG.");
    img.src = src;
  }

  function onFile(f: File | undefined) {
    if (!f) return;
    if (!f.type.startsWith("image/")) return setError("Please choose an image file.");
    const url = URL.createObjectURL(f);
    load(url, f.name);
  }

  const v = res ? VERDICT[res.kind] : null;
  const colour = res
    ? res.blueFrac > 0.5
      ? "Mostly blue — little soot. Typical of diffusion flames in microgravity or weakly burning flames."
      : res.yellowFrac > 0.5
        ? "Mostly yellow/orange — glowing soot. Soot radiates heat, which can help a larger flame lose energy (as NASA's Flame Design experiment saw), but also means more smoke."
        : "Mixed colours — part blue, part luminous."
    : "";
  const matches = res ? MATCHES[res.kind].map((id) => INVESTIGATIONS.find((i) => i.id === id)).filter(Boolean) : [];

  return (
    <div className="grid gap-6 lg:grid-cols-[1fr_1fr]">
      <section className="rounded-2xl border border-line bg-surface p-5">
        <h2 className="text-base font-semibold text-ink">Flame image check</h2>
        <p className="mt-1 text-sm text-ink-2">
          Upload a photo of a flame. The image is analysed in your browser — it isn&apos;t uploaded anywhere.
        </p>
        <label
          className="mt-4 flex cursor-pointer flex-col items-center justify-center rounded-xl border border-dashed border-line bg-surface-2 p-6 text-center text-sm text-ink-2 hover:border-ink-3"
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
            onFile(e.dataTransfer.files?.[0]);
          }}
        >
          <span className="text-ink">Drop an image here or click to choose</span>
          <span className="mt-1 text-xs text-ink-3">JPG or PNG · works best on a dark background</span>
          <input type="file" accept="image/*" className="hidden" onChange={(e) => onFile(e.target.files?.[0])} />
        </label>
        <div className="mt-3 flex flex-wrap items-center gap-2 text-xs">
          <span className="text-ink-3">Or try an example (illustration):</span>
          {SAMPLES.map((s) => (
            <button key={s.src} onClick={() => load(s.src, s.label)} className="rounded-full border border-line px-3 py-1 text-ink-2 hover:border-ink-3 hover:text-ink">
              {s.label}
            </button>
          ))}
        </div>
        {error && <p className="mt-3 text-sm text-crit">{error}</p>}
        <div className="mt-4 overflow-hidden rounded-xl bg-bg">
          <canvas ref={canvasRef} className={`mx-auto block max-h-[320px] max-w-full ${res ? "" : "hidden"}`} />
          {!res && <div className="grid h-[200px] place-items-center text-sm text-ink-3">No image yet</div>}
        </div>
        {res && <p className="mt-2 text-xs text-ink-3">{name} — detected flame highlighted, dashed box = flame extent.</p>}
      </section>

      <section className="rounded-2xl border border-line bg-surface p-5">
        {!res || !v ? (
          <div className="text-sm text-ink-3">
            <p className="text-ink-2">What it checks:</p>
            <ul className="mt-2 list-disc space-y-1 pl-5">
              <li>Shape — round (microgravity) vs tall teardrop (Earth gravity) vs flat spreading front</li>
              <li>Colour — blue (little soot) vs yellow/orange (glowing soot)</li>
              <li>Which NASA PSI investigations studied that kind of flame</li>
            </ul>
          </div>
        ) : (
          <>
            <h3 className="text-lg font-semibold text-ink">{v.title}</h3>
            <p className="mt-2 text-sm leading-relaxed text-ink-2">{v.body}</p>
            {res.kind !== "none" && (
              <>
                <p className="mt-3 text-sm text-ink-2">
                  <span className="text-ink">Colour: </span>
                  {colour}
                  {res.hollow && res.kind === "microgravity" && " The centre is darker than the edge — a thin flame shell around the fuel, as in spherical diffusion flames."}
                </p>
                <dl className="mt-4 grid grid-cols-2 gap-2 text-xs sm:grid-cols-4">
                  {[
                    ["Height ÷ width", res.aspect.toFixed(2)],
                    ["Tip ÷ widest part", res.taper.toFixed(2)],
                    ["Blue pixels", `${(res.blueFrac * 100).toFixed(0)}%`],
                    ["Yellow/orange", `${(res.yellowFrac * 100).toFixed(0)}%`],
                  ].map(([k, val]) => (
                    <div key={k} className="rounded-lg bg-surface-2 p-2">
                      <dt className="text-ink-3">{k}</dt>
                      <dd className="font-mono text-ink">{val}</dd>
                    </div>
                  ))}
                </dl>
              </>
            )}
            {matches.length > 0 && (
              <div className="mt-5">
                <div className="text-sm font-semibold text-ink">{res.kind === "earth" ? "How NASA's microgravity experiments differ" : "NASA investigations that studied this"}</div>
                <ul className="mt-2 space-y-2">
                  {matches.map((inv) => (
                    <li key={inv!.id} className="rounded-lg bg-surface-2 p-3 text-sm">
                      <a className="font-medium text-accent hover:underline" href={inv!.source_url} target="_blank" rel="noreferrer">
                        {inv!.name} ↗
                      </a>
                      <p className="mt-1 text-xs text-ink-2">{inv!.key_findings[0]}</p>
                    </li>
                  ))}
                </ul>
              </div>
            )}
            <p className="mt-4 text-[11px] text-ink-3">
              Rule-based image measurement (shape and colour thresholds), not a trained classifier — lighting, camera exposure and cropping affect it. Cool flames are invisible to normal cameras.
            </p>
          </>
        )}
      </section>
    </div>
  );
}
