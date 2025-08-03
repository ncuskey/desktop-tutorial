export type Cell = 0 | 1;

export interface Ant {
  x: number;
  y: number;
}

/** Create a grid filled with sand (1) except the top row which is empty. */
export function createGrid(width: number, height: number): Cell[][] {
  return Array.from({ length: height }, (_, y) =>
    Array.from({ length: width }, () => (y === 0 ? 0 : 1))
  );
}


/**
 * Advance the simulation by one step, moving ants and carving tunnels.
 */
export function step(grid: Cell[][], ants: Ant[]): void {
  const height = grid.length;
  const width = grid[0].length;

  for (let i = ants.length - 1; i >= 0; i--) {
    const ant = ants[i];
    const moves: Array<[number, number]> = [[0, 1], [-1, 1], [1, 1]];
    if (Math.random() < 0.5) {
      [moves[1], moves[2]] = [moves[2], moves[1]];
    }

    let moved = false;
    for (const [dx, dy] of moves) {
      const nx = ant.x + dx;
      const ny = ant.y + dy;
      if (nx < 0 || nx >= width || ny >= height) {
        continue;
      }
      if (grid[ny][nx] === 1) {
        grid[ny][nx] = 0; // dig through sand
        ant.x = nx;
        ant.y = ny;
        moved = true;
        break;
      } else if (grid[ny][nx] === 0) {
        ant.x = nx;
        ant.y = ny;
        moved = true;
        break;
      }
    }

    if (!moved || ant.y >= height - 1) {
      // remove ants that cannot move or reached bottom
      ants.splice(i, 1);
    }
  }
}

