import { useState, type FormEvent } from "react";
import type { EnrollResponse } from "../api";
import { client } from "../client";
import ImageInput from "../components/ImageInput";
import { releaseImageItems, type ImageItem } from "../lib/images";

const MAX_IMAGES = 10;

interface Props {
  onEnrolled: () => void;
}

export default function Enroll({ onEnrolled }: Props) {
  const [name, setName] = useState("");
  const [images, setImages] = useState<ImageItem[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<{ res: EnrollResponse; images: ImageItem[] } | null>(null);

  const count = images.length;
  const nudge =
    count === 0
      ? "Add 3 to 5 clear, front-facing photos of the same person."
      : count < 3
        ? `${count} selected. 3 to 5 photos give a more reliable template.`
        : count > 5
          ? `${count} selected. 3 to 5 is usually enough; extra photos add little.`
          : `${count} selected. Good.`;
  const nudgeTone = count === 0 ? "" : count >= 3 && count <= 5 ? "ok" : "warn";

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!name.trim() || count === 0) return;
    setBusy(true);
    setError(null);
    try {
      const res = await client.enroll(name.trim(), images.map((i) => i.blob));
      if (result) releaseImageItems(result.images);
      setResult({ res, images });
      setImages([]);
      setName("");
      onEnrolled();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="page-grid">
      <form className="card" onSubmit={submit}>
        <h2>Enroll a person</h2>
        <p className="muted">
          Each photo is checked for a face and screened for deepfakes. Accepted photos are averaged into one face template.
        </p>

        <label className="field">
          <span className="field-label">Name</span>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. Jane Doe"
            autoComplete="off"
            disabled={busy}
            required
          />
        </label>

        <div className="field">
          <span className="field-label">
            Face photos <span className="muted">({count}/{MAX_IMAGES})</span>
          </span>
          <ImageInput images={images} onChange={setImages} max={MAX_IMAGES} disabled={busy} />
          <div className={`hint ${nudgeTone}`}>{nudge}</div>
        </div>

        {error && (
          <div className="alert alert-error" role="alert">
            <strong>Enrollment failed.</strong> {error}
          </div>
        )}

        <div className="actions">
          <button type="submit" className="btn btn-primary" disabled={busy || !name.trim() || count === 0}>
            {busy ? "Enrolling…" : "Enroll"}
          </button>
        </div>
      </form>

      <div>
        {result ? (
          <EnrollResult res={result.res} images={result.images} />
        ) : (
          <div className="card placeholder">
            <div className="placeholder-title">No enrollment yet</div>
            <p className="muted">The result appears here: how many photos were used and why any were rejected.</p>
          </div>
        )}
      </div>
    </div>
  );
}

function EnrollResult({ res, images }: { res: EnrollResponse; images: ImageItem[] }) {
  const rejected = new Map(res.rejected.map((r) => [r.index, r.reason]));
  return (
    <section className="card" aria-live="polite">
      <div className="decision decision-ok compact">
        <div>
          <div className="decision-title">Enrolled {res.name}</div>
          <div className="decision-detail">
            Identity #{res.identity_id} · {res.images_used} of {images.length} photo{images.length === 1 ? "" : "s"} used
          </div>
        </div>
      </div>
      <ul className="enroll-list">
        {images.map((img, i) => {
          const reason = rejected.get(i);
          return (
            <li key={img.id} className={reason ? "rejected" : "accepted"}>
              <img src={img.url} alt={`Photo ${i + 1}`} />
              <div>
                <div className="enroll-item-title">
                  Photo {i + 1}
                  <span className={`tag ${reason ? "tag-bad" : "tag-ok"}`}>{reason ? "Rejected" : "Used"}</span>
                </div>
                <div className="muted small">{reason ?? img.label}</div>
              </div>
            </li>
          );
        })}
      </ul>
      {res.rejected
        .filter((r) => r.index >= images.length)
        .map((r) => (
          <div key={r.index} className="muted small">
            Image {r.index + 1}: {r.reason}
          </div>
        ))}
    </section>
  );
}
