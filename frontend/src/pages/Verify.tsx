import { useState, type FormEvent } from "react";
import type { VerifyResponse } from "../api";
import { client } from "../client";
import ImageInput from "../components/ImageInput";
import ResultCard from "../components/ResultCard";
import type { ImageItem } from "../lib/images";
import type { IdentitiesState } from "../lib/useIdentities";

interface Props {
  identities: IdentitiesState;
  onGoEnroll: () => void;
}

export default function Verify({ identities, onGoEnroll }: Props) {
  const [selected, setSelected] = useState<string>("");
  const [images, setImages] = useState<ImageItem[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<{ res: VerifyResponse; url: string; name: string } | null>(null);

  const list = identities.items;
  // Fall back to the first identity if nothing (or a since-deleted one) is selected.
  const current = list.find((i) => String(i.identity_id) === selected) ?? list[0];
  const probe = images[0];

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!current || !probe) return;
    setBusy(true);
    setError(null);
    try {
      const res = await client.verify(current.identity_id, probe.blob);
      // Own URL for the result so removing/replacing the input thumbnail does not break it.
      const url = URL.createObjectURL(probe.blob);
      setResult((prev) => {
        if (prev) URL.revokeObjectURL(prev.url);
        return { res, url, name: current.name };
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="page-grid">
      <form className="card" onSubmit={submit}>
        <h2>Verify a face</h2>
        <p className="muted">
          Choose the claimed identity and submit one photo. It is accepted only if it is judged real and matches the
          enrolled template.
        </p>

        <label className="field">
          <span className="field-label">Claimed identity</span>
          {list.length > 0 ? (
            <select
              value={current ? String(current.identity_id) : ""}
              onChange={(e) => setSelected(e.target.value)}
              disabled={busy}
            >
              {list.map((i) => (
                <option key={i.identity_id} value={i.identity_id}>
                  {i.name}
                </option>
              ))}
            </select>
          ) : (
            <div className="empty-inline">
              {identities.loading ? (
                "Loading identities…"
              ) : identities.error ? (
                <>Could not load identities: {identities.error}</>
              ) : (
                <>
                  No one is enrolled yet.{" "}
                  <button type="button" className="link" onClick={onGoEnroll}>
                    Enroll someone first
                  </button>
                  .
                </>
              )}
            </div>
          )}
        </label>

        <div className="field">
          <span className="field-label">Probe photo</span>
          <ImageInput images={images} onChange={setImages} max={1} disabled={busy} />
        </div>

        {error && (
          <div className="alert alert-error" role="alert">
            <strong>Verification failed.</strong> {error}
          </div>
        )}

        <div className="actions">
          <button type="submit" className="btn btn-primary" disabled={busy || !current || !probe}>
            {busy ? "Verifying…" : "Verify"}
          </button>
        </div>
      </form>

      <div>
        {result ? (
          <ResultCard result={result.res} imageUrl={result.url} identityName={result.name} />
        ) : (
          <div className="card placeholder">
            <div className="placeholder-title">No result yet</div>
            <p className="muted">
              The decision appears here with the similarity and fake-probability scores against their thresholds, and the
              detected face outlined on the photo.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
