import React, { useRef, useEffect, useState } from 'react';
import type { Mesh, State } from '../worker';

interface Props {
  mesh?: Mesh;
  states?: State[];
}

/**
 * Canvas component for rendering the map mesh and state boundaries.
 */
const MapCanvas: React.FC<Props> = ({ mesh, states }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [transform, setTransform] = useState({ x: 0, y: 0, scale: 1 });
  const dragging = useRef(false);
  const lastPos = useRef({ x: 0, y: 0 });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !mesh) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const { width, height } = canvas;
    ctx.save();
    ctx.clearRect(0, 0, width, height);
    ctx.translate(transform.x, transform.y);
    ctx.scale(transform.scale, transform.scale);

    // Draw triangles
    for (let i = 0; i < mesh.triangles.length; i += 3) {
      const a = mesh.triangles[i] * 2;
      const b = mesh.triangles[i + 1] * 2;
      const c = mesh.triangles[i + 2] * 2;
      ctx.beginPath();
      ctx.moveTo(mesh.points[a], mesh.points[a + 1]);
      ctx.lineTo(mesh.points[b], mesh.points[b + 1]);
      ctx.lineTo(mesh.points[c], mesh.points[c + 1]);
      ctx.closePath();
      ctx.strokeStyle = '#999';
      ctx.stroke();
    }

    // TODO: draw states properly
    if (states) {
      ctx.strokeStyle = '#ff0000';
      states.forEach((state) => {
        const cells = state.cellIndices;
        ctx.beginPath();
        cells.forEach((id, idx) => {
          const idx2 = id * 2;
          if (idx === 0) ctx.moveTo(mesh.points[idx2], mesh.points[idx2 + 1]);
          else ctx.lineTo(mesh.points[idx2], mesh.points[idx2 + 1]);
        });
        ctx.closePath();
        ctx.stroke();
      });
    }
    ctx.restore();
  }, [mesh, states, transform]);

  const onWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY < 0 ? 1.1 : 0.9;
    setTransform((t) => ({ ...t, scale: t.scale * delta }));
  };

  const onMouseDown = (e: React.MouseEvent) => {
    dragging.current = true;
    lastPos.current = { x: e.clientX, y: e.clientY };
  };

  const onMouseMove = (e: React.MouseEvent) => {
    if (!dragging.current) return;
    const dx = e.clientX - lastPos.current.x;
    const dy = e.clientY - lastPos.current.y;
    lastPos.current = { x: e.clientX, y: e.clientY };
    setTransform((t) => ({ ...t, x: t.x + dx, y: t.y + dy }));
  };

  const onMouseUp = () => {
    dragging.current = false;
  };

  return (
    <canvas
      ref={canvasRef}
      width={800}
      height={600}
      onWheel={onWheel}
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
      onMouseUp={onMouseUp}
      // TODO: style via CSS
    />
  );
};

export default MapCanvas;
