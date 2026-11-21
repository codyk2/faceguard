// Single entry point for backend calls. Pages import `client`, never `api` directly,
// so the dev-only mock can be swapped in with VITE_MOCK=1 without touching api.ts.
import { api } from "./api";
import { mockApi } from "./mock";

export const IS_MOCK = import.meta.env.VITE_MOCK === "1";

export const client: typeof api = IS_MOCK ? mockApi : api;
