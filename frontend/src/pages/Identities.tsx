import { useState } from "react";
import { client } from "../client";
import type { IdentitiesState } from "../lib/useIdentities";

interface Props {
  identities: IdentitiesState;
  onGoEnroll: () => void;
}

function formatDate(iso: string): string {
  const d = new Date(iso);
  return Number.isNaN(d.getTime())
    ? iso
    : d.toLocaleString(undefined, { year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

export default function Identities({ identities, onGoEnroll }: Props) {
  const { items, loading, error, reload } = identities;
  const [confirming, setConfirming] = useState<number | null>(null);
  const [deleting, setDeleting] = useState<number | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  const remove = async (id: number) => {
    setDeleting(id);
    setDeleteError(null);
    try {
      await client.deleteIdentity(id);
      setConfirming(null);
      await reload();
    } catch (err) {
      setDeleteError(err instanceof Error ? err.message : String(err));
    } finally {
      setDeleting(null);
    }
  };

  return (
    <section className="card">
      <div className="card-head">
        <div>
          <h2>Enrolled identities</h2>
          <p className="muted">
            {items.length} {items.length === 1 ? "identity" : "identities"} stored. Deleting removes the face template.
          </p>
        </div>
        <button type="button" className="btn btn-secondary" onClick={() => void reload()} disabled={loading}>
          {loading ? "Refreshing…" : "Refresh"}
        </button>
      </div>

      {error && (
        <div className="alert alert-error" role="alert">
          <strong>Could not load identities.</strong> {error}
        </div>
      )}
      {deleteError && (
        <div className="alert alert-error" role="alert">
          <strong>Delete failed.</strong> {deleteError}
        </div>
      )}

      {items.length === 0 ? (
        !loading &&
        !error && (
          <div className="empty">
            <div className="placeholder-title">No identities yet</div>
            <button type="button" className="btn btn-primary" onClick={onGoEnroll}>
              Enroll a person
            </button>
          </div>
        )
      ) : (
        <div className="table-wrap">
          <table className="table">
            <thead>
              <tr>
                <th className="num">ID</th>
                <th>Name</th>
                <th className="num">Images used</th>
                <th>Enrolled</th>
                <th className="actions-col" aria-label="Actions" />
              </tr>
            </thead>
            <tbody>
              {items.map((i) => (
                <tr key={i.identity_id}>
                  <td className="num muted">{i.identity_id}</td>
                  <td className="strong">{i.name}</td>
                  <td className="num">{i.images_used}</td>
                  <td className="muted">{formatDate(i.created_at)}</td>
                  <td className="actions-col">
                    {confirming === i.identity_id ? (
                      <span className="confirm">
                        <span className="muted small">Delete {i.name}?</span>
                        <button
                          type="button"
                          className="btn btn-danger btn-sm"
                          onClick={() => void remove(i.identity_id)}
                          disabled={deleting !== null}
                        >
                          {deleting === i.identity_id ? "Deleting…" : "Delete"}
                        </button>
                        <button
                          type="button"
                          className="btn btn-ghost btn-sm"
                          onClick={() => setConfirming(null)}
                          disabled={deleting !== null}
                        >
                          Cancel
                        </button>
                      </span>
                    ) : (
                      <button type="button" className="btn btn-ghost btn-sm" onClick={() => setConfirming(i.identity_id)}>
                        Delete
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
