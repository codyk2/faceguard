export interface ImageItem {
  id: string;
  blob: Blob;
  url: string;
  label: string;
}

let counter = 0;

export function makeImageItem(blob: Blob, label: string): ImageItem {
  counter += 1;
  return { id: `img-${Date.now()}-${counter}`, blob, url: URL.createObjectURL(blob), label };
}

export function releaseImageItems(items: ImageItem[]): void {
  items.forEach((i) => URL.revokeObjectURL(i.url));
}

export function formatPercent(v: number, digits = 1): string {
  return `${(v * 100).toFixed(digits)}%`;
}
