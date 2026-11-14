# Deploy (both parts from this one repo)

## 1. Backend → Render (free)
New → Web Service → connect this repo.
- Root Directory: `backend`
- Runtime: Python 3
- Build Command: `pip install -r requirements.txt && python scripts/generate_data.py && python scripts/import_nasa.py && python scripts/train_model.py`
- Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Environment variable: `PYTHON_VERSION` = `3.12.8`
- Health check path: `/health`

Copy the URL Render gives you (e.g. `https://flame-backend.onrender.com`) and open `/health` — it should say `"status":"ok"`.

## 2. Frontend → Vercel
Add New → Project → import this repo.
- Root Directory: `frontend`
- Environment variables:
  - `BACKEND_URL` = the Render URL (no trailing slash)
  - `ANTHROPIC_API_KEY` = Claude API key
Deploy. The badge top-right should say **● Live model · trained on synthetic + NASA data**.

If you change `BACKEND_URL` later: Settings → Environment Variables → update → Deployments → Redeploy.
Render's free tier sleeps when idle — open the site ~1 minute before demoing.
