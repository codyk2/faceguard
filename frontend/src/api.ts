// Typed client. FROZEN: mirrors backend/app/schemas.py exactly.

export type Decision = "ACCEPT" | "REJECT_FAKE" | "REJECT_MISMATCH" | "NO_FACE";

export interface FaceBox { x1: number; y1: number; x2: number; y2: number; score: number }
export interface Thresholds { t_sim: number; t_fake: number }
export interface HealthResponse { status: string; models_loaded: boolean; device: string; verifier: string; detector: string }
export interface RejectedImage { index: number; reason: string }
export interface EnrollResponse { identity_id: number; name: string; images_used: number; rejected: RejectedImage[] }
export interface Identity { identity_id: number; name: string; images_used: number; created_at: string }
export interface VerifyResponse {
  decision: Decision;
  similarity: number | null;
  fake_probability: number | null;
  thresholds: Thresholds;
  face_box: FaceBox | null;
  warnings: string[];
  timings_ms: Record<string, number>;
}
export interface DetectResponse { fake_probability: number | null; label: string | null; face_box: FaceBox | null; warnings: string[] }

const BASE = "/api";

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail ?? detail; } catch { /* not json */ }
    if (typeof detail === "string") throw new Error(detail);
    const d = detail as { message?: string; rejected?: RejectedImage[] };
    if (d && d.message) {
      const reasons = (d.rejected ?? []).map(r => `image ${r.index + 1}: ${r.reason}`).join("; ");
      throw new Error(reasons ? `${d.message} (${reasons})` : d.message);
    }
    throw new Error(JSON.stringify(detail));
  }
  return res.status === 204 ? (undefined as T) : res.json();
}

export const api = {
  health: () => fetch(`${BASE}/health`).then(handle<HealthResponse>),
  identities: () => fetch(`${BASE}/identities`).then(handle<Identity[]>),
  deleteIdentity: (id: number) => fetch(`${BASE}/identities/${id}`, { method: "DELETE" }).then(handle<void>),
  enroll: (name: string, files: Blob[]) => {
    const fd = new FormData();
    fd.append("name", name);
    files.forEach((f, i) => fd.append("files", f, (f as File).name ?? `img${i}.jpg`));
    return fetch(`${BASE}/enroll`, { method: "POST", body: fd }).then(handle<EnrollResponse>);
  },
  verify: (identityId: number, file: Blob) => {
    const fd = new FormData();
    fd.append("identity_id", String(identityId));
    fd.append("file", file, (file as File).name ?? "probe.jpg");
    return fetch(`${BASE}/verify`, { method: "POST", body: fd }).then(handle<VerifyResponse>);
  },
  detect: (file: Blob) => {
    const fd = new FormData();
    fd.append("file", file, (file as File).name ?? "probe.jpg");
    return fetch(`${BASE}/detect`, { method: "POST", body: fd }).then(handle<DetectResponse>);
  },
};
