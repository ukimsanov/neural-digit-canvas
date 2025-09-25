'use client';

import React, { useRef, useEffect, useState, forwardRef, useImperativeHandle } from 'react';
import { cn } from '@/lib/utils';

interface DrawingCanvasProps {
  className?: string;
  onDrawingChange?: () => void;
}

export interface DrawingCanvasRef {
  clear: () => void;
  getCanvas: () => HTMLCanvasElement | null;
  isEmpty: () => boolean;
}

// Helper function to calculate line width based on canvas size and thickness setting
const getLineWidth = (canvasSize: number, thickness: number): number => {
  const baseWidth = canvasSize / 20;
  const multiplier = 0.5 + (thickness - 1) * 0.3; // Thickness 1-5 maps to 0.5x-1.7x multiplier
  return baseWidth * multiplier;
};

const DrawingCanvas = forwardRef<DrawingCanvasRef, DrawingCanvasProps>(
  ({ className, onDrawingChange }, ref) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [isDrawing, setIsDrawing] = useState(false);
    const [context, setContext] = useState<CanvasRenderingContext2D | null>(null);
    const [isEmpty, setIsEmpty] = useState(true);
    const [brushThickness, setBrushThickness] = useState(3); // Default thickness level (1-5)

    useEffect(() => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      // Set canvas size
      const size = Math.min(window.innerWidth - 32, 400);
      canvas.width = size;
      canvas.height = size;

      // Configure drawing style
      ctx.fillStyle = 'white';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = 'black';
      ctx.lineWidth = getLineWidth(size, brushThickness);
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      setContext(ctx);
    }, [brushThickness]);

    // Update line width when brush thickness changes
    useEffect(() => {
      if (context && canvasRef.current) {
        context.lineWidth = getLineWidth(canvasRef.current.width, brushThickness);
      }
    }, [context, brushThickness]);

    useImperativeHandle(ref, () => ({
      clear: () => {
        if (context && canvasRef.current) {
          context.fillStyle = 'white';
          context.fillRect(0, 0, canvasRef.current.width, canvasRef.current.height);
          setIsEmpty(true);
          onDrawingChange?.();
        }
      },
      getCanvas: () => canvasRef.current,
      isEmpty: () => isEmpty,
    }));

    const startDrawing = (e: React.MouseEvent | React.TouchEvent) => {
      if (!context || !canvasRef.current) return;

      setIsDrawing(true);
      setIsEmpty(false);

      const rect = canvasRef.current.getBoundingClientRect();
      const x = 'touches' in e
        ? e.touches[0].clientX - rect.left
        : e.nativeEvent.offsetX;
      const y = 'touches' in e
        ? e.touches[0].clientY - rect.top
        : e.nativeEvent.offsetY;

      context.beginPath();
      context.moveTo(x, y);
    };

    const draw = (e: React.MouseEvent | React.TouchEvent) => {
      if (!isDrawing || !context || !canvasRef.current) return;

      const rect = canvasRef.current.getBoundingClientRect();
      const x = 'touches' in e
        ? e.touches[0].clientX - rect.left
        : e.nativeEvent.offsetX;
      const y = 'touches' in e
        ? e.touches[0].clientY - rect.top
        : e.nativeEvent.offsetY;

      context.lineTo(x, y);
      context.stroke();
      onDrawingChange?.();
    };

    const stopDrawing = () => {
      if (!context) return;
      setIsDrawing(false);
      context.closePath();
    };

    return (
      <div className={cn('relative flex flex-col items-center', className)}>
        {/* Brush thickness control */}
        <div className="mb-4 flex items-center gap-3 p-3 bg-white/10 backdrop-blur-sm rounded-xl border border-white/20 shadow-lg">
          <label htmlFor="brush-thickness" className="text-sm font-medium text-gray-700 whitespace-nowrap">
            Brush Size:
          </label>
          <input
            id="brush-thickness"
            type="range"
            min="1"
            max="5"
            step="1"
            value={brushThickness}
            onChange={(e) => setBrushThickness(Number(e.target.value))}
            className="w-20 h-2 bg-gradient-to-r from-blue-200 to-blue-400 rounded-lg appearance-none cursor-pointer slider"
            aria-label="Adjust brush thickness"
          />
          <span className="text-sm text-gray-600 min-w-[20px] text-center font-mono">
            {brushThickness}
          </span>
        </div>

        <canvas
          ref={canvasRef}
          className="border-4 border-blue-500 rounded-2xl cursor-crosshair touch-none bg-white shadow-lg max-w-full"
          onMouseDown={startDrawing}
          onMouseMove={draw}
          onMouseUp={stopDrawing}
          onMouseLeave={stopDrawing}
          onTouchStart={startDrawing}
          onTouchMove={draw}
          onTouchEnd={stopDrawing}
          role="img"
          aria-label={isEmpty ? "Empty drawing canvas for digit input" : "Canvas with drawn digit"}
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              // Focus on canvas for drawing
            }
          }}
        />
        <div className="mt-4 text-center text-xs sm:text-sm text-gray-500 leading-tight">
          Draw a digit (0-9)
        </div>
      </div>
    );
  }
);

DrawingCanvas.displayName = 'DrawingCanvas';

export default DrawingCanvas;