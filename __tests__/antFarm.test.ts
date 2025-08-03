import { createGrid, step, Ant } from '../src/antFarm';

describe('ant farm simulation', () => {
  it('allows an ant to dig downward through sand', () => {
    const grid = createGrid(3, 3);
    const ants: Ant[] = [{ x: 1, y: 0 }];
    step(grid, ants);
    expect(grid[1][1]).toBe(0);
    expect(ants[0].y).toBe(1);
  });
});

