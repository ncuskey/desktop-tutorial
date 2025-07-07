// Map generation WebWorker

// Import algorithms from mapgen4
import { makeMesh } from './mesh';
import Map from './map';

let map: Map | null = null;

// TODO: consider using WebAssembly versions for performance

/** Dual mesh representation. */
export interface Mesh {
  points: Float32Array;
  triangles: Uint32Array;
  regions: Uint32Array;
  isBoundary: Uint8Array;
  elevationT: Float32Array;
  elevationR: Float32Array;
}

/** Cell from parsed map data. */
export interface Cell { id: number; x: number; y: number; }

/** State grouping cells. */
export interface State { id: number; name: string; color: string; cellIndices: number[]; capitalIndex?: number; }

/** Town location. */
export interface Burg { id: number; cellId: number; x: number; y: number; name: string; }

/** Road connection between towns. */
export interface Road { id: number; fromIndex: number; toIndex: number; type: string; }

/** Map JSON structure from FMG. */
export interface MapData { settings?: any; cells: Cell[]; states?: State[]; burgs?: Burg[]; roads?: Road[]; }

/** Options controlling state generation. */
export interface StateOptions { count: number; }

/** Options controlling road generation. */
export interface RoadOptions { maxDistance: number; }

/** Elevation algorithm parameters. */
export interface ElevationParams {
  noisyCoastlines: number;
  hillHeight: number;
  mountainSharpness: number;
  oceanDepth: number;
}

/** Global generation constraints. */
export interface Constraints { size: number; constraints: Float32Array; }

/** River generation parameters. */
export interface RiverParams { flow: number; minFlow: number; riverWidth: number; }

/** Generate the Delaunay/Voronoi mesh and pick peak triangles. */
export async function generateMesh(): Promise<{ mesh: Mesh; peaks: number[] }> {
  const { mesh, t_peaks } = await makeMesh();
  if (mesh.triangles.length % 3 !== 0) {
    throw new Error('Triangle list must be a multiple of 3');
  }
  map = new Map(mesh, t_peaks);
  return { mesh, peaks: t_peaks };
}

/** Assign elevation across triangles and regions. */
export async function assignElevation(
  _data: { mesh: Mesh; peaks: number[] },
  params: ElevationParams,
  constraints: Constraints,
): Promise<{ elevationT: Float32Array; elevationR: Float32Array }> {
  if (!map) throw new Error('Mesh not generated');
  const mesh = (map as any).mesh as Mesh;
  if (mesh.triangles.length % 3 !== 0) throw new Error('Invalid mesh');
  map.assignElevation(params, constraints);
  return { elevationT: mesh.elevationT, elevationR: mesh.elevationR };
}

/** Compute river flow on the mesh. */
export async function assignRivers(
  _data: { mesh: Mesh },
  params: RiverParams,
): Promise<{ flowT: Float32Array }> {
  if (!map) throw new Error('Mesh not generated');
  const mesh = (map as any).mesh as Mesh;
  if (mesh.triangles.length % 3 !== 0) throw new Error('Invalid mesh');
  map.assignRivers(params);
  const flowT = (mesh as any).flowT as Float32Array;
  return { flowT };
}

// Stubs for FMG integration
export async function loadMapJSON(path: string): Promise<MapData> {
  const res = await fetch(path);
  return res.json() as Promise<MapData>;
}

export async function generateStates(cells: Cell[], options: StateOptions): Promise<State[]> {
  if (!cells.length) throw new Error('cells empty');
  const count = Math.max(1, Math.min(options.count, cells.length));
  const states: State[] = [];
  for (let i = 0; i < count; i++) {
    states.push({ id: i, name: `State ${i}`, color: '#000000', cellIndices: [], capitalIndex: undefined });
  }
  cells.forEach((c, i) => { states[i % count].cellIndices.push(c.id); });
  states.forEach(s => { s.capitalIndex = s.cellIndices[0]; });
  return states;
}

export async function generateRoads(burgs: Burg[], params: RoadOptions): Promise<Road[]> {
  if (burgs.length < 2) return [];
  const roads: Road[] = [];
  let id = 0;
  for (let i = 0; i < burgs.length; i++) {
    for (let j = i + 1; j < burgs.length; j++) {
      const a = burgs[i];
      const b = burgs[j];
      const dx = a.x - b.x;
      const dy = a.y - b.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist <= params.maxDistance) {
        roads.push({ id: id++, fromIndex: a.id, toIndex: b.id, type: 'road' });
      }
    }
  }
  return roads;
}

/**
 * Handle incoming messages.
 * @param event Worker message containing command and payload
 */
onmessage = async (event: MessageEvent) => {
  const { cmd, payload } = event.data;
  try {
    switch (cmd) {
      case 'generateMesh': {
        /**
         * Payload: undefined
         * Response: { mesh: Mesh, peaks: number[] }
         */
        const result = await generateMesh();
        postMessage({ cmd: 'generateMesh', data: result });
        break;
      }
      case 'assignElevation': {
        /**
         * Payload: { data: { mesh: Mesh; peaks: number[] }, params: ElevationParams, constraints: Constraints }
         * Response: { elevationT: Float32Array, elevationR: Float32Array }
         */
        const { data, params, constraints } = payload as {
          data: { mesh: Mesh; peaks: number[] };
          params: ElevationParams;
          constraints: Constraints;
        };
        const result = await assignElevation(data, params, constraints);
        postMessage({ cmd: 'assignElevation', data: result });
        break;
      }
      case 'assignRivers': {
        /**
         * Payload: { data: { mesh: Mesh }, params: RiverParams }
         * Response: { flowT: Float32Array }
         */
        const { data, params } = payload as {
          data: { mesh: Mesh };
          params: RiverParams;
        };
        const result = await assignRivers(data, params);
        postMessage({ cmd: 'assignRivers', data: result });
        break;
      }
      case 'loadMapJSON': {
        /**
         * Payload: string path to .map JSON
         * Response: MapData parsed from file
         */
        const path = payload as string;
        const result = await loadMapJSON(path);
        postMessage({ cmd: 'loadMapJSON', data: result });
        break;
      }
      case 'generateStates': {
        /**
         * Payload: { cells: Cell[]; options: StateOptions }
         * Response: State[]
         */
        const { cells, options } = payload as {
          cells: Cell[];
          options: StateOptions;
        };
        const result = await generateStates(cells, options);
        postMessage({ cmd: 'generateStates', data: result });
        break;
      }
      case 'generateRoads': {
        /**
         * Payload: { burgs: Burg[]; params: RoadOptions }
         * Response: Road[]
         */
        const { burgs, params } = payload as {
          burgs: Burg[];
          params: RoadOptions;
        };
        const result = await generateRoads(burgs, params);
        postMessage({ cmd: 'generateRoads', data: result });
        break;
      }
      default:
        postMessage({ cmd: `${cmd}Error`, error: 'Unknown command' });
    }
  } catch (error: any) {
    postMessage({ cmd: `${cmd}Error`, error: error.message ?? String(error) });
  }
};

