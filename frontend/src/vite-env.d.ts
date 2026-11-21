/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** "1" swaps the real backend for the in-memory mock in src/mock.ts. */
  readonly VITE_MOCK?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
