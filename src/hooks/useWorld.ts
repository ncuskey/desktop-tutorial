import { useEffect, useState, useCallback } from 'react';
import type {
  Mesh, MapData, State, Road, Cell, Burg,
  StateOptions, RoadOptions,
  ElevationParams, Constraints, RiverParams
} from '../../worker';
import {
  generateMesh, assignElevation, assignRivers,
  loadMapJSON, generateStates, generateRoads
} from '../workerClient';
import {
  initializeLore, applyOutcome, generateAdventureHooks
} from '../loreClient';

// Define combined status type
type Status = 'loading' | 'ready' | 'updatingHooks' | 'error';

export interface AdventureHook { /* … */ }
export interface WorldState { /* … */ }
export interface WorldLore { /* … */ }
export interface QuestOutcome { questID: string; success: boolean; }

const STORAGE_KEY = 'world-state';

export function useWorld() {
  // Load initial from localStorage
  const initialWorld = (() => {
    const raw = typeof localStorage !== 'undefined' ? localStorage.getItem(STORAGE_KEY) : null;
    if (raw) {
      try {
        return JSON.parse(raw) as WorldState & { lore?: WorldLore };
      } catch {}
    }
    return { hooks: [] } as WorldState;
  })();

  const [world, setWorld] = useState<WorldState>(initialWorld);
  const [lore, setLore] = useState<WorldLore | undefined>(initialWorld.lore);
  const [status, setStatus] = useState<Status>(
    initialWorld.mesh ? 'ready' : 'loading'
  );
  const [error, setError] = useState<string | null>(null);

  // Persist
  useEffect(() => {
    try {
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({ ...world, lore, hooks: world.hooks })
      );
    } catch {}
  }, [world, lore]);

  const initWorld = useCallback(async () => {
    setStatus('loading');
    try {
      // mesh → elevation → rivers → mapData → states → roads
      const { mesh, peaks } = await generateMesh();
      setWorld(w => ({ ...w, mesh, peaks }));

      const elev = await assignElevation(
        { mesh, peaks },
        { noisyCoastlines: 0.5, hillHeight: 0.3, mountainSharpness: 0.8, oceanDepth: 1.0 },
        { size: 1024, constraints: new Float32Array() }
      );
      setWorld(w => ({ ...w, elevationT: elev.elevationT, elevationR: elev.elevationR }));

      const rivers = await assignRivers({ mesh }, { flow: 1, minFlow: 0.1, riverWidth: 1 });
      setWorld(w => ({ ...w, flowT: rivers.flowT }));

      const mapData = await loadMapJSON('/maps/sample.map.json');
      setWorld(w => ({ ...w, mapData }));

      const states = await generateStates(mapData.cells as Cell[], { count: 5 } as StateOptions);
      setWorld(w => ({ ...w, states }));

      const roads = await generateRoads(mapData.burgs as Burg[], { maxDistance: 50 } as RoadOptions);
      setWorld(w => ({ ...w, roads }));

      // **Lore initialization**
      const initialLore = await initializeLore(states);
      setLore(initialLore);
      const initialHooks = await generateAdventureHooks(initialLore);
      setWorld(w => ({ ...w, hooks: initialHooks }));

      setStatus('ready');
    } catch (err: any) {
      console.error(err);
      setError(err.message ?? String(err));
      setStatus('error');
    }
  }, []);

  useEffect(() => {
    if (!world.mesh) {
      initWorld();
    }
  }, [world.mesh, initWorld]);

  const completeHook = useCallback(async (id: string, success: boolean) => {
    setStatus('updatingHooks');
    try {
      const outcome: QuestOutcome = { questID: id, success };
      const updatedLore = await applyOutcome(outcome);
      setLore(updatedLore);
      const newHooks = await generateAdventureHooks(updatedLore);
      setWorld(w => ({ ...w, hooks: newHooks }));
      setStatus('ready');
    } catch (err: any) {
      console.error(err);
      setError(err.message ?? String(err));
      setStatus('error');
    }
  }, []);

  return { world, lore, status, error, completeHook };
}