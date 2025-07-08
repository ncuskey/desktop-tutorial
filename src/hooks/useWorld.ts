import { useEffect, useState, useCallback } from 'react';
import type {
  Mesh,
  MapData,
  State,
  Road,
  Cell,
  Burg,
  StateOptions,
  RoadOptions,
  ElevationParams,
  Constraints,
  RiverParams,
} from '../../worker';
import {
  generateMesh,
  assignElevation,
  assignRivers,
  loadMapJSON,
  generateStates,
  generateRoads,
} from '../workerClient';

/** Adventure hook description. */
export interface AdventureHook {
  id: string;
  description: string;
  completed?: boolean;
}

/** Shape of the world state stored by the application. */
export interface WorldState {
  mesh?: Mesh;
  peaks?: number[];
  elevationT?: Float32Array;
  elevationR?: Float32Array;
  flowT?: Float32Array;
  mapData?: MapData;
  states?: State[];
  roads?: Road[];
  hooks: AdventureHook[];
}

type Status = 'loading' | 'ready' | 'error';

const STORAGE_KEY = 'world-state';

/**
 * Custom hook managing world generation and persistence.
 */
export function useWorld() {
  const initialWorld: WorldState = (() => {
    const raw = typeof localStorage !== 'undefined' ? localStorage.getItem(STORAGE_KEY) : null;
    if (raw) {
      try {
        const parsed: WorldState = JSON.parse(raw);
        return { ...parsed, hooks: parsed.hooks || [] };
      } catch {
        // TODO: better error handling
      }
    }
    return { hooks: [] };
  })();

  const [world, setWorld] = useState<WorldState>(initialWorld);

  const [status, setStatus] = useState<Status>(initialWorld.mesh ? 'ready' : 'loading');
  const [error, setError] = useState<string | null>(null);

  // Persist world whenever it changes
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({ ...world, hooks: world.hooks }));
    } catch {
      // TODO: better error handling
    }
  }, [world]);

  const initWorld = useCallback(async () => {
    setStatus('loading');
    setError(null);
    try {
      const { mesh, peaks } = await generateMesh();
      setWorld((w) => ({ ...w, mesh, peaks }));

      const elevationParams: ElevationParams = {
        noisyCoastlines: 0.5,
        hillHeight: 0.3,
        mountainSharpness: 0.8,
        oceanDepth: 1.0,
      };
      const constraints: Constraints = { size: 1024, constraints: new Float32Array() };
      const elev = await assignElevation({ mesh, peaks }, elevationParams, constraints);
      setWorld((w) => ({ ...w, elevationT: elev.elevationT, elevationR: elev.elevationR }));

      const riverParams: RiverParams = { flow: 1, minFlow: 0.1, riverWidth: 1 };
      const rivers = await assignRivers({ mesh }, riverParams);
      setWorld((w) => ({ ...w, flowT: rivers.flowT }));

      const mapData = await loadMapJSON('/maps/sample.map.json');
      setWorld((w) => ({ ...w, mapData }));

      const states = await generateStates(mapData.cells as Cell[], { count: 5 } as StateOptions);
      setWorld((w) => ({ ...w, states }));

      const roads = await generateRoads(mapData.burgs as Burg[], { maxDistance: 50 } as RoadOptions);
      setWorld((w) => ({ ...w, roads }));
      setStatus('ready');
    } catch (err) {
      console.error(err);
      setStatus('error');
      setError(err instanceof Error ? err.message : String(err));
    }
  }, []);

  useEffect(() => {
    if (!world.mesh) {
      initWorld();
    }
  }, [world.mesh, initWorld]);

  const completeHook = useCallback((id: string, success: boolean) => {
    // TODO: send to LoreEngine worker when integrated
    setWorld((w) => ({
      ...w,
      hooks: w.hooks.map((h) => (h.id === id ? { ...h, completed: success } : h)),
    }));
  }, []);

  return { world, status, error, completeHook };
}
