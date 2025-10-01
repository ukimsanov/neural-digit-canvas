import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function canvasToBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => {
        if (blob) {
          resolve(blob);
        } else {
          reject(new Error('Failed to convert canvas to blob'));
        }
      },
      'image/png'
    );
  });
}

export function preprocessCanvas(canvas: HTMLCanvasElement): HTMLCanvasElement {
  const processedCanvas = document.createElement('canvas');
  processedCanvas.width = 28;
  processedCanvas.height = 28;
  const ctx = processedCanvas.getContext('2d');

  if (!ctx) {
    throw new Error('Could not get canvas context');
  }

  // Fill with black background (MNIST uses black background)
  ctx.fillStyle = 'black';
  ctx.fillRect(0, 0, 28, 28);

  // Draw the original canvas scaled down
  ctx.filter = 'invert(1)'; // Invert colors if needed
  ctx.drawImage(canvas, 0, 0, 28, 28);

  return processedCanvas;
}

export function canvasToBase64(canvas: HTMLCanvasElement): string {
  const dataURL = canvas.toDataURL('image/png');
  return dataURL.split(',')[1];
}