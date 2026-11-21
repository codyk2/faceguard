import { useState } from "react";
import type { Decision, FaceBox, VerifyResponse } from "../api";
import { formatPercent } from "../lib/images";

interface Props {
  result: VerifyResponse;
  imageUrl: string;
  identityName?: string;
}

const DECISIONS: Record<Decision, { title: string; detail: string; tone: string }> = {
  ACCEPT: { title: "Accepted", detail: "Real face that matches the claimed identity.", tone: "ok" },
  REJECT_FAKE: { title: "Rejected: suspected fake", detail: "The deepfake screen flagged this image.", tone: "bad" },
  REJECT_MISMATCH: { title: "Rejected: identity mismatch", detail: "The face does not match the enrolled template.", tone: "warn" },
  NO_FACE: { title: "No face detected", detail: "No usable face was found in the image.", tone: "neutral" },
};

export default function ResultCard({ result, imageUrl, identityName }: Props) {
  const d = DECISIONS[result.decision];
  const { t_sim, t_fake } = result.thresholds;

  return (
    <section className={`card result tone-${d.tone}`} aria-live="polite">
      <div className={`decision decision-${d.tone}`}>
        <DecisionIcon tone={d.tone} />
        <div>
          <div className="decision-code">{result.decision}</div>
          <div className="decision-title">{d.title}</div>
          <div className="decision-detail">
            {d.detail}
            {identityName && <> Claimed identity: <strong>{identityName}</strong>.</>}
          </div>
        </div>
      </div>

      <div className="result-body">
        <ProbeImage url={imageUrl} box={result.face_box} tone={d.tone} />

        <div className="metrics">
          <ScoreBar
            label="Similarity"
            hint="cosine vs. enrolled template"
            value={result.similarity}
            threshold={t_sim}
            thresholdLabel="T_SIM"
            pass={result.similarity !== null && result.similarity >= t_sim}
            format={(v) => v.toFixed(3)}
            rule={`pass if ≥ ${t_sim.toFixed(2)}`}
          />
          <ScoreBar
            label="Fake probability"
            hint="ViT deepfake classifier"
            value={result.fake_probability}
            threshold={t_fake}
            thresholdLabel="T_FAKE"
            pass={result.fake_probability !== null && result.fake_probability < t_fake}
            format={(v) => formatPercent(v)}
            rule={`pass if < ${formatPercent(t_fake, 0)}`}
          />

          {result.warnings.length > 0 && (
            <div className="warnings">
              <div className="section-label">Warnings</div>
              <ul>
                {result.warnings.map((w, i) => (
                  <li key={i}>{w}</li>
                ))}
              </ul>
            </div>
          )}

          <Timings timings={result.timings_ms} />
        </div>
      </div>
    </section>
  );
}

function ScoreBar(props: {
  label: string;
  hint: string;
  value: number | null;
  threshold: number;
  thresholdLabel: string;
  pass: boolean;
  format: (v: number) => string;
  rule: string;
}) {
  const { label, hint, value, threshold, thresholdLabel, pass, format, rule } = props;
  const clamp = (v: number) => Math.min(1, Math.max(0, v));
  const fill = value === null ? 0 : clamp(value) * 100;
  const marker = clamp(threshold) * 100;

  return (
    <div className="score">
      <div className="score-head">
        <div>
          <span className="score-label">{label}</span>
          <span className="muted small"> · {hint}</span>
        </div>
        <span className={`score-value ${value === null ? "na" : pass ? "pass" : "fail"}`}>
          {value === null ? "n/a" : format(value)}
        </span>
      </div>
      <div
        className="bar"
        role="meter"
        aria-label={label}
        aria-valuemin={0}
        aria-valuemax={1}
        aria-valuenow={value ?? undefined}
      >
        {value !== null && <div className={`bar-fill ${pass ? "pass" : "fail"}`} style={{ width: `${fill}%` }} />}
        <div className="bar-marker" style={{ left: `${marker}%` }} />
        <div className="bar-marker-label" style={{ left: `${marker}%` }}>
          {thresholdLabel} {format(threshold)}
        </div>
      </div>
      <div className="score-foot muted small">
        <span>0</span>
        <span>{value === null ? "not computed (no face)" : rule}</span>
        <span>1</span>
      </div>
    </div>
  );
}

function ProbeImage({ url, box, tone }: { url: string; box: FaceBox | null; tone: string }) {
  const [natural, setNatural] = useState<{ w: number; h: number; url: string } | null>(null);
  const size = natural && natural.url === url ? natural : null;

  // Box coordinates are in the image's natural pixel space; expressing them as
  // percentages of the natural size scales them to whatever size the image is displayed at.
  const style =
    box && size
      ? {
          left: `${(box.x1 / size.w) * 100}%`,
          top: `${(box.y1 / size.h) * 100}%`,
          width: `${((box.x2 - box.x1) / size.w) * 100}%`,
          height: `${((box.y2 - box.y1) / size.h) * 100}%`,
        }
      : null;

  return (
    <figure className="probe">
      <div className="probe-frame">
        <img
          src={url}
          alt="Probe"
          onLoad={(e) => setNatural({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight, url })}
        />
        {style && box && (
          <div className={`face-box face-box-${tone}`} style={style}>
            <span className="face-box-label">face {formatPercent(box.score, 0)}</span>
          </div>
        )}
      </div>
      <figcaption className="muted small">
        {box ? `Face box (${Math.round(box.x1)}, ${Math.round(box.y1)}) to (${Math.round(box.x2)}, ${Math.round(box.y2)}) px` : "No face box returned"}
      </figcaption>
    </figure>
  );
}

function Timings({ timings }: { timings: Record<string, number> }) {
  const entries = Object.entries(timings);
  if (entries.length === 0) return null;
  const total = timings.total;
  const steps = entries.filter(([k]) => k !== "total");
  return (
    <div className="timings">
      <div className="section-label">Timings</div>
      <dl>
        {steps.map(([k, v]) => (
          <div key={k}>
            <dt>{k}</dt>
            <dd>{v.toFixed(1)} ms</dd>
          </div>
        ))}
        {total !== undefined && (
          <div className="timings-total">
            <dt>total</dt>
            <dd>{total.toFixed(1)} ms</dd>
          </div>
        )}
      </dl>
    </div>
  );
}

function DecisionIcon({ tone }: { tone: string }) {
  const common = { width: 28, height: 28, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 2.2, strokeLinecap: "round" as const, strokeLinejoin: "round" as const, "aria-hidden": true };
  if (tone === "ok")
    return (
      <svg {...common}>
        <circle cx="12" cy="12" r="10" />
        <path d="M7.5 12.5l3 3 6-6.5" />
      </svg>
    );
  if (tone === "neutral")
    return (
      <svg {...common}>
        <circle cx="12" cy="12" r="10" />
        <path d="M12 8v4.5M12 16h.01" />
      </svg>
    );
  return (
    <svg {...common}>
      <circle cx="12" cy="12" r="10" />
      <path d="M8.5 8.5l7 7M15.5 8.5l-7 7" />
    </svg>
  );
}
