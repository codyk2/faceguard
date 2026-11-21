// Dev-only in-memory stand-in for the backend. Enabled with VITE_MOCK=1 (see client.ts).
//
// Verify outcome is picked from the probe file name when it contains one of
// "accept" | "fake" | "mismatch" | "noface"; otherwise successive calls cycle
// ACCEPT -> REJECT_FAKE -> REJECT_MISMATCH -> NO_FACE so every state is reachable.
// Enroll rejects files whose name contains "fake" or "noface", and with 5+ images
// rejects the last one as too small. Enrolling an existing name fails like a 409.
import type {
  api,
  Decision,
  DetectResponse,
  EnrollResponse,
  FaceBox,
  HealthResponse,
  Identity,
  RejectedImage,
  VerifyResponse,
} from "./api";

const T_SIM = 0.35;
const T_FAKE = 0.5;

let nextId = 4;
const identities: Identity[] = [
  { identity_id: 1, name: "George W Bush", images_used: 5, created_at: "2026-11-20T15:42:10Z" },
  { identity_id: 2, name: "Serena Williams", images_used: 4, created_at: "2026-11-20T16:03:55Z" },
  { identity_id: 3, name: "Tony Blair", images_used: 3, created_at: "2026-11-21T09:18:31Z" },
];

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const fileName = (b: Blob) => ((b as File).name ?? "").toLowerCase();

async function imageSize(b: Blob): Promise<{ w: number; h: number }> {
  try {
    const bmp = await createImageBitmap(b);
    const size = { w: bmp.width, h: bmp.height };
    bmp.close();
    return size;
  } catch {
    return { w: 640, h: 480 };
  }
}

async function centeredBox(b: Blob): Promise<FaceBox> {
  const { w, h } = await imageSize(b);
  const side = Math.min(w, h) * 0.5;
  const cx = w / 2;
  const cy = h * 0.46;
  return {
    x1: Math.round(cx - side / 2),
    y1: Math.round(cy - side * 0.6),
    x2: Math.round(cx + side / 2),
    y2: Math.round(cy + side * 0.6),
    score: 0.87,
  };
}

const ORDER: Decision[] = ["ACCEPT", "REJECT_FAKE", "REJECT_MISMATCH", "NO_FACE"];
let cycle = 0;

function pickDecision(b: Blob): Decision {
  const n = fileName(b);
  if (n.includes("noface")) return "NO_FACE";
  if (n.includes("mismatch")) return "REJECT_MISMATCH";
  if (n.includes("fake")) return "REJECT_FAKE";
  if (n.includes("accept")) return "ACCEPT";
  return ORDER[cycle++ % ORDER.length];
}

function fail(message: string): never {
  throw new Error(message);
}

export const mockApi: typeof api = {
  async health(): Promise<HealthResponse> {
    await sleep(150);
    return {
      status: "ok",
      models_loaded: true,
      device: "cpu",
      verifier: "buffalo_l (mock)",
      detector: "Deep-Fake-Detector-v2 (mock)",
    };
  },

  async identities(): Promise<Identity[]> {
    await sleep(200);
    return identities.map((i) => ({ ...i }));
  },

  async deleteIdentity(id: number): Promise<void> {
    await sleep(200);
    const idx = identities.findIndex((i) => i.identity_id === id);
    if (idx < 0) fail("Identity not found");
    identities.splice(idx, 1);
  },

  async enroll(name: string, files: Blob[]): Promise<EnrollResponse> {
    await sleep(700);
    const trimmed = name.trim();
    if (!trimmed) fail("name must not be empty");
    if (files.length === 0) fail("at least one image is required");
    if (identities.some((i) => i.name.toLowerCase() === trimmed.toLowerCase())) {
      fail(`identity '${trimmed}' already exists`);
    }
    const rejected: RejectedImage[] = [];
    files.forEach((f, index) => {
      const n = fileName(f);
      if (n.includes("noface")) rejected.push({ index, reason: "no face detected" });
      else if (n.includes("fake")) rejected.push({ index, reason: "image judged fake (p_fake=0.93)" });
    });
    if (files.length >= 5 && !rejected.some((r) => r.index === files.length - 1)) {
      rejected.push({ index: files.length - 1, reason: "face too small (42 px < 60 px)" });
    }
    const used = files.length - rejected.length;
    if (used === 0) fail("no usable face images; all were rejected");
    const identity: Identity = {
      identity_id: nextId++,
      name: trimmed,
      images_used: used,
      created_at: new Date().toISOString(),
    };
    identities.push(identity);
    return { identity_id: identity.identity_id, name: identity.name, images_used: used, rejected };
  },

  async verify(identityId: number, file: Blob): Promise<VerifyResponse> {
    await sleep(600);
    if (!identities.some((i) => i.identity_id === identityId)) fail("Identity not found");
    const decision = pickDecision(file);
    const thresholds = { t_sim: T_SIM, t_fake: T_FAKE };
    const timings_ms = { decode: 4.1, detect: 38.6, embed: 21.3, fake: 64.9, total: 131.2 };
    if (decision === "NO_FACE") {
      return {
        decision,
        similarity: null,
        fake_probability: null,
        thresholds,
        face_box: null,
        warnings: ["no face detected in probe image"],
        timings_ms: { decode: 4.0, detect: 35.2, total: 39.5 },
      };
    }
    const face_box = await centeredBox(file);
    const values: Record<Exclude<Decision, "NO_FACE">, [number, number, string[]]> = {
      ACCEPT: [0.71, 0.06, []],
      REJECT_FAKE: [0.64, 0.91, ["fake probability above threshold"]],
      REJECT_MISMATCH: [0.12, 0.08, ["multiple faces detected; using the largest"]],
    };
    const [similarity, fake_probability, warnings] = values[decision];
    return { decision, similarity, fake_probability, thresholds, face_box, warnings, timings_ms };
  },

  async detect(file: Blob): Promise<DetectResponse> {
    await sleep(400);
    const fake = fileName(file).includes("fake");
    return {
      fake_probability: fake ? 0.91 : 0.06,
      label: fake ? "fake" : "real",
      face_box: await centeredBox(file),
      warnings: [],
    };
  },
};
