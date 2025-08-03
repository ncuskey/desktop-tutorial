import React, { useRef, useEffect } from 'react';
import { createGrid, step, Ant } from './antFarm';

const WIDTH = 120;
const HEIGHT = 80;
const SCALE = 4; // size of each cell in pixels

/** Canvas component that simulates ants digging tunnels through sand. */
const AntFarm: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const gridRef = useRef(createGrid(WIDTH, HEIGHT));
  const antsRef = useRef<Ant[]>([]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    function draw() {
      const grid = gridRef.current;

      // Draw sand
      ctx.fillStyle = '#deb887';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Clear empty cells
      ctx.fillStyle = '#ffffff';
      for (let y = 0; y < HEIGHT; y++) {
        for (let x = 0; x < WIDTH; x++) {
          if (grid[y][x] === 0) {
            ctx.clearRect(x * SCALE, y * SCALE, SCALE, SCALE);
          }
        }
      }

      // Draw ants
      ctx.fillStyle = '#000000';
      for (const ant of antsRef.current) {
        ctx.fillRect(ant.x * SCALE, ant.y * SCALE, SCALE, SCALE);
      }
    }

    function tick() {
      if (antsRef.current.length < 50) {
        antsRef.current.push({ x: Math.floor(Math.random() * WIDTH), y: 0 });
      }
      step(gridRef.current, antsRef.current);
      draw();
      requestAnimationFrame(tick);
    }

    tick();
  }, []);

  return (
    <canvas
      ref={canvasRef}
      width={WIDTH * SCALE}
      height={HEIGHT * SCALE}
      style={{ border: '1px solid #000' }}
    />
  );
};

export default AntFarm;

