import { useCallback, useEffect, useRef, useState } from "react";
import { makeImageItem, type ImageItem } from "../lib/images";

interface Props {
  images: ImageItem[];
  onChange: (images: ImageItem[]) => void;
  /** Maximum number of images. 1 means a single image that is replaced on each add. */
  max?: number;
  disabled?: boolean;
}

type Mode = "upload" | "camera";

export default function ImageInput({ images, onChange, max = 1, disabled = false }: Props) {
  const [mode, setMode] = useState<Mode>("upload");
  const multiple = max > 1;
  const full = multiple && images.length >= max;

  const add = useCallback(
    (blobs: { blob: Blob; label: string }[]) => {
      if (blobs.length === 0) return;
      if (!multiple) {
        images.forEach((i) => URL.revokeObjectURL(i.url));
        const last = blobs[blobs.length - 1];
        onChange([makeImageItem(last.blob, last.label)]);
        return;
      }
      const room = Math.max(0, max - images.length);
      const added = blobs.slice(0, room).map((b) => makeImageItem(b.blob, b.label));
      onChange([...images, ...added]);
    },
    [images, max, multiple, onChange],
  );

  const remove = (id: string) => {
    const item = images.find((i) => i.id === id);
    if (item) URL.revokeObjectURL(item.url);
    onChange(images.filter((i) => i.id !== id));
  };

  return (
    <div className="image-input">
      <div className="segmented" role="tablist" aria-label="Image source">
        <button
          type="button"
          role="tab"
          aria-selected={mode === "upload"}
          className={mode === "upload" ? "active" : ""}
          onClick={() => setMode("upload")}
          disabled={disabled}
        >
          Upload file{multiple ? "s" : ""}
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={mode === "camera"}
          className={mode === "camera" ? "active" : ""}
          onClick={() => setMode("camera")}
          disabled={disabled}
        >
          Webcam
        </button>
      </div>

      {mode === "upload" ? (
        <UploadZone multiple={multiple} disabled={disabled || full} onFiles={add} />
      ) : (
        <CameraCapture disabled={disabled || full} onCapture={add} />
      )}

      {images.length > 0 && (
        <ul className="thumbs">
          {images.map((img, i) => (
            <li key={img.id} className="thumb">
              <img src={img.url} alt={`Selected image ${i + 1}`} />
              {multiple && <span className="thumb-index">{i + 1}</span>}
              <button
                type="button"
                className="thumb-remove"
                onClick={() => remove(img.id)}
                disabled={disabled}
                aria-label={`Remove image ${i + 1}`}
                title="Remove"
              >
                ×
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function UploadZone({
  multiple,
  disabled,
  onFiles,
}: {
  multiple: boolean;
  disabled: boolean;
  onFiles: (files: { blob: Blob; label: string }[]) => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const accept = (list: FileList | null) => {
    if (!list) return;
    const files = Array.from(list).filter((f) => f.type.startsWith("image/"));
    onFiles(files.map((f) => ({ blob: f, label: f.name })));
  };

  return (
    <div
      className={`dropzone${dragging ? " dragging" : ""}${disabled ? " disabled" : ""}`}
      onClick={() => !disabled && inputRef.current?.click()}
      onKeyDown={(e) => {
        if (!disabled && (e.key === "Enter" || e.key === " ")) {
          e.preventDefault();
          inputRef.current?.click();
        }
      }}
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        if (!disabled) accept(e.dataTransfer.files);
      }}
      role="button"
      tabIndex={disabled ? -1 : 0}
      aria-disabled={disabled}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        multiple={multiple}
        hidden
        onChange={(e) => {
          accept(e.target.files);
          e.target.value = "";
        }}
      />
      <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true">
        <path d="M12 16V4m0 0l-4 4m4-4l4 4" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2" strokeLinecap="round" />
      </svg>
      <div className="dropzone-text">
        {disabled ? (
          "Image limit reached"
        ) : (
          <>
            <strong>Click to choose</strong> or drag {multiple ? "images" : "an image"} here
          </>
        )}
      </div>
      <div className="muted small">JPEG or PNG</div>
    </div>
  );
}

function CameraCapture({
  disabled,
  onCapture,
}: {
  disabled: boolean;
  onCapture: (files: { blob: Blob; label: string }[]) => void;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [ready, setReady] = useState(false);
  const shots = useRef(0);

  useEffect(() => {
    let stream: MediaStream | null = null;
    let cancelled = false;

    if (!navigator.mediaDevices?.getUserMedia) {
      setError("Camera is not available in this browser (a secure context such as localhost is required).");
      return;
    }
    navigator.mediaDevices
      .getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" }, audio: false })
      .then((s) => {
        if (cancelled) {
          s.getTracks().forEach((t) => t.stop());
          return;
        }
        stream = s;
        const video = videoRef.current;
        if (video) {
          video.srcObject = s;
          video.play().catch(() => undefined);
        }
      })
      .catch((e: unknown) => {
        if (cancelled) return;
        const name = e instanceof DOMException ? e.name : "";
        setError(
          name === "NotAllowedError"
            ? "Camera permission was denied. Allow access in the browser, or upload a file instead."
            : name === "NotFoundError"
              ? "No camera was found on this device."
              : "Could not start the camera.",
        );
      });

    return () => {
      cancelled = true;
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  const capture = () => {
    const video = videoRef.current;
    if (!video || !video.videoWidth) return;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(
      (blob) => {
        if (!blob) return;
        shots.current += 1;
        const file = new File([blob], `webcam-${shots.current}.jpg`, { type: "image/jpeg" });
        onCapture([{ blob: file, label: file.name }]);
      },
      "image/jpeg",
      0.92,
    );
  };

  if (error) return <div className="camera camera-error">{error}</div>;

  return (
    <div className="camera">
      <video
        ref={videoRef}
        playsInline
        muted
        onLoadedMetadata={() => setReady(true)}
        aria-label="Webcam preview"
      />
      {!ready && <div className="camera-status muted">Starting camera…</div>}
      <button type="button" className="btn btn-secondary camera-shoot" onClick={capture} disabled={!ready || disabled}>
        Capture photo
      </button>
    </div>
  );
}
