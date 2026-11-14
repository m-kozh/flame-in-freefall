import type { NextConfig } from "next";

// BACKEND_URL = the FastAPI service (e.g. https://burn-boundary-api.onrender.com).
// Requests to /backend/* are proxied server-side, so the browser never calls the backend
// directly — no CORS setup needed on the backend.
const BACKEND_URL = process.env.BACKEND_URL?.replace(/\/$/, "");

const nextConfig: NextConfig = {
  env: {
    NEXT_PUBLIC_BACKEND_ENABLED: BACKEND_URL ? "1" : "",
  },
  async rewrites() {
    return BACKEND_URL ? [{ source: "/backend/:path*", destination: `${BACKEND_URL}/:path*` }] : [];
  },
};

export default nextConfig;
