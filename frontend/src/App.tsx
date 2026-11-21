import { useEffect, useRef, useState } from "react";
import type { HealthResponse } from "./api";
import { client, IS_MOCK } from "./client";
import { useIdentities } from "./lib/useIdentities";
import Enroll from "./pages/Enroll";
import Identities from "./pages/Identities";
import Verify from "./pages/Verify";

type View = "enroll" | "verify" | "identities";
const VIEWS: { id: View; label: string }[] = [
  { id: "enroll", label: "Enroll" },
  { id: "verify", label: "Verify" },
  { id: "identities", label: "Identities" },
];

function viewFromHash(): View {
  const h = window.location.hash.replace(/^#\/?/, "");
  return VIEWS.some((v) => v.id === h) ? (h as View) : "verify";
}

type Health = { state: "checking" } | { state: "online"; data: HealthResponse } | { state: "offline"; error: string };

function useHealth(intervalMs = 10000): Health {
  const [health, setHealth] = useState<Health>({ state: "checking" });
  useEffect(() => {
    let alive = true;
    const check = () =>
      client
        .health()
        .then((data) => alive && setHealth({ state: "online", data }))
        .catch((e: unknown) => alive && setHealth({ state: "offline", error: e instanceof Error ? e.message : String(e) }));
    check();
    const t = window.setInterval(check, intervalMs);
    return () => {
      alive = false;
      window.clearInterval(t);
    };
  }, [intervalMs]);
  return health;
}

function HealthIndicator({ health }: { health: Health }) {
  if (health.state === "checking")
    return (
      <div className="health health-checking">
        <span className="dot" /> Connecting…
      </div>
    );
  if (health.state === "offline")
    return (
      <div className="health health-offline" title={health.error}>
        <span className="dot" /> Backend offline
      </div>
    );
  const { data } = health;
  const title = `status: ${data.status}\nverifier: ${data.verifier}\ndetector: ${data.detector}`;
  return (
    <div className={`health ${data.models_loaded ? "health-online" : "health-loading"}`} title={title}>
      <span className="dot" />
      {data.models_loaded ? "Models loaded" : "Models loading"}
      <span className="health-device">{data.device}</span>
    </div>
  );
}

export default function App() {
  const [view, setView] = useState<View>(viewFromHash);
  const health = useHealth();
  const identities = useIdentities();
  const { reload: reloadIdentities } = identities;

  // Refetch identities when the backend comes back after being offline.
  const wasOffline = useRef(false);
  useEffect(() => {
    if (health.state === "online" && wasOffline.current) void reloadIdentities();
    wasOffline.current = health.state === "offline";
  }, [health.state, reloadIdentities]);

  useEffect(() => {
    const onHash = () => setView(viewFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const go = (v: View) => {
    window.location.hash = `/${v}`;
    setView(v);
  };

  return (
    <div className="app">
      <header className="topbar">
        <div className="topbar-inner">
          <div className="brand">
            <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
              <path d="M12 2.5l8 3v6c0 5-3.4 8.8-8 10-4.6-1.2-8-5-8-10v-6l8-3z" strokeLinejoin="round" />
              <circle cx="12" cy="10.5" r="2.6" />
              <path d="M7.8 16.5c.9-1.9 2.4-2.8 4.2-2.8s3.3.9 4.2 2.8" strokeLinecap="round" />
            </svg>
            <div>
              <div className="brand-name">FaceGuard</div>
              <div className="brand-sub">Face verification with deepfake screening</div>
            </div>
          </div>
          <nav className="tabs" aria-label="Views">
            {VIEWS.map((v) => (
              <button
                key={v.id}
                type="button"
                className={view === v.id ? "active" : ""}
                aria-current={view === v.id ? "page" : undefined}
                onClick={() => go(v.id)}
              >
                {v.label}
                {v.id === "identities" && identities.items.length > 0 && (
                  <span className="count">{identities.items.length}</span>
                )}
              </button>
            ))}
          </nav>
          <div className="topbar-right">
            {IS_MOCK && <span className="mock-tag" title="VITE_MOCK=1: responses come from src/mock.ts">mock</span>}
            <HealthIndicator health={health} />
          </div>
        </div>
      </header>

      {health.state === "offline" && (
        <div className="banner" role="status">
          Cannot reach the backend at <code>/api</code>. Start it on port 8000, or run the UI with{" "}
          <code>VITE_MOCK=1</code> to use mock data.
        </div>
      )}

      <main className="main">
        {view === "enroll" && <Enroll onEnrolled={() => void identities.reload()} />}
        {view === "verify" && <Verify identities={identities} onGoEnroll={() => go("enroll")} />}
        {view === "identities" && <Identities identities={identities} onGoEnroll={() => go("enroll")} />}
      </main>
    </div>
  );
}
