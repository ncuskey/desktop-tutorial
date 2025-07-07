import type { Mesh, Cell, State, Burg, Road, MapData, StateOptions, RoadOptions, ElevationParams, Constraints, RiverParams } from '../worker';

/**
 * Wrapper around the map generation WebWorker. Provides promise based
 * accessors for each command defined in `worker.ts`.
 */
class WorkerClient {
  private worker: Worker;
  private pending: Record<string, Array<{resolve: (data: any) => void; reject: (err: Error) => void}>> = {};

  constructor() {
    this.worker = new Worker(new URL('../worker.ts', import.meta.url));
    this.worker.onmessage = this.handleMessage.bind(this);
  }

  private handleMessage(event: MessageEvent) {
    const { cmd, data, error } = event.data as { cmd: string; data?: any; error?: string };
    const baseCmd = cmd.replace(/Error$/, '');
    const queue = this.pending[baseCmd];
    if (!queue || queue.length === 0) return;
    const { resolve, reject } = queue.shift()!;
    if (cmd.endsWith('Error')) {
      reject(new Error(error));
    } else {
      resolve(data);
    }
  }

  private call<T>(cmd: string, payload: any): Promise<T> {
    return new Promise((resolve, reject) => {
      if (!this.pending[cmd]) this.pending[cmd] = [];
      this.pending[cmd].push({ resolve, reject });
      this.worker.postMessage({ cmd, payload });
    });
  }

  /** Generate the mesh and peak triangles. */
  generateMesh(config?: { pointCount: number }): Promise<{ mesh: Mesh; peaks: number[] }> {
    // Worker currently ignores config
    return this.call<{ mesh: Mesh; peaks: number[] }>('generateMesh', config);
  }

  /** Assign elevation across the mesh. */
  assignElevation(data: { mesh: Mesh; peaks: number[] }, params: ElevationParams, constraints: Constraints): Promise<{ elevationT: Float32Array; elevationR: Float32Array }> {
    return this.call<{ elevationT: Float32Array; elevationR: Float32Array }>('assignElevation', { data, params, constraints });
  }

  /** Compute river flow. */
  assignRivers(data: { mesh: Mesh }, params: RiverParams): Promise<{ flowT: Float32Array }> {
    return this.call<{ flowT: Float32Array }>('assignRivers', { data, params });
  }

  /** Load an exported FMG map JSON file. */
  loadMapJSON(path: string): Promise<MapData> {
    return this.call<MapData>('loadMapJSON', path);
  }

  /** Generate states from cells. */
  generateStates(cells: Cell[], options: StateOptions): Promise<State[]> {
    return this.call<State[]>('generateStates', { cells, options });
  }

  /** Generate roads between burgs. */
  generateRoads(burgs: Burg[], params: RoadOptions): Promise<Road[]> {
    return this.call<Road[]>('generateRoads', { burgs, params });
  }
}

const workerClient = new WorkerClient();

export const generateMesh = workerClient.generateMesh.bind(workerClient);
export const assignElevation = workerClient.assignElevation.bind(workerClient);
export const assignRivers = workerClient.assignRivers.bind(workerClient);
export const loadMapJSON = workerClient.loadMapJSON.bind(workerClient);
export const generateStates = workerClient.generateStates.bind(workerClient);
export const generateRoads = workerClient.generateRoads.bind(workerClient);

export default workerClient;
