# Burn Boundary — NASA Space Apps 2026 (Flame in Freefall)

Which materials go from safe to dangerous when the atmosphere changes from ISS air (21% O₂, 14.7 psi)
to Moon-base air (≥30% O₂, ≤10.2 psi) — how sure we are, and which experiment NASA should run next.

## Run it
```bash
npm install
cp .env.example .env.local   # add ANTHROPIC_API_KEY for the chat tab
npm run dev                  # http://localhost:3000
```
With `BACKEND_URL` empty the app uses `src/lib/mock.ts` and shows a "Mock model" badge.
Set `BACKEND_URL` and everything switches over — no other code changes.

Badge states: **Mock model** (no backend set) · **Connecting** · **● Live model** · **Model offline — showing demo data**
(backend set but failing; the site keeps working on the mock instead of breaking).

## Deploy (Vercel)
1. Push this repo to GitHub → vercel.com → Add New Project → import the repo (framework: Next.js, no settings to change).
2. Project → Settings → Environment Variables:
   - `BACKEND_URL` = the backend's public URL, e.g. `https://burn-boundary-api.onrender.com`
   - `ANTHROPIC_API_KEY` = your Claude API key
   - optional `CLAUDE_MODEL`, `CHAT_RATE_LIMIT`
3. **Redeploy after changing `BACKEND_URL`** — it is baked in at build time.
4. In the Anthropic console, set a monthly spend limit on the key.

How the backend connection works: the browser calls `/backend/...` on our own domain and Next.js forwards it
to `BACKEND_URL` (see `next.config.ts`). The backend doesn't need CORS and its URL never has to change in code.

Free backend hosts (Render) sleep when idle and take ~30–50 s to wake. Open the site a minute before demoing,
or ping `BACKEND_URL/health` every 10 minutes with a free uptime monitor.

## Screens
1. **Atmosphere shift** — habitat switch, flammability map (burn risk / model uncertainty), material verdicts ISS → habitat, recommended next experiments
2. **Predict** — sliders → `/predict`, confidence intervals, physics vs learned, SHAP from `/explain`
3. **Evidence (PSI)** — the PSI investigations ranked by relevance to the chosen habitat
4. **Ask AI** — Claude answers grounded in `src/data/investigations.json`, citing investigations

## API contract (backend must match `src/lib/types.ts`)
- `GET /health` → `{"status":"ok"}`
- `GET /materials` → `{"materials":[{"id","name","seen_in"}]}` (optional; frontend falls back to its own list)
- `POST /predict` `{o2, pressure_kpa, flow_cm_s, material}` → spread rate + CI, extinction prob + CI, physics baseline, in_data_range
- `GET /map?material=&pressure_kpa=` → x (O₂ %), y (airflow cm/s), z[y][x] extinction prob, uncertainty, tested_points
- `GET /suggest?material=` → points where the model is least certain
- `GET /explain` → feature importances
If the backend doesn't serve `/materials`, its material ids must match `cotton`, `sibal`, `pmma`, `nomex`.

## Who owns what
- Frontend: `src/app`, `src/components`
- Data: `src/data/investigations.json` — verify every entry against PSI, add PSI IDs + DOIs ("Cite this Investigation")
- Backend: `../backend` (FastAPI, LightGBM residual ensemble) — deployed separately (Render). The frontend uses its `/map`, `/predict`, `/suggest`, `/explain`, `/materials` endpoints through the `/backend` proxy.

## Before submitting
- Replace mock tested points with real PSI test-matrix data (they are labelled MOCK in tooltips)
- Confirm the rover atmosphere preset (34% / 8.2 psi is an assumption)
- Chat is rate-limited per IP (`CHAT_RATE_LIMIT`, default 20 per 10 min) and inputs are capped at 1,000 characters
- AI-use disclosure: chat uses the Claude API; code written with AI assistance
