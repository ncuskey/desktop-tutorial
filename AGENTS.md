# Worker API Documentation

This repository includes a WebWorker (`worker.ts`) that provides map generation and manipulation utilities for the browser. Frontend code can send commands to the worker via `postMessage` and receive results asynchronously via `onmessage` events.

## Commands

### `generateMesh`

**Payload**
```ts
undefined // no payload required
```

**Response**
```ts
{ mesh: Mesh; peaks: number[] }
```

**Error**
```ts
{ cmd: 'generateMeshError', error: string }
```

Generates a Delaunay/Voronoi mesh and identifies peak triangles. Must be called before running elevation or river algorithms.

### `assignElevation`

**Payload**
```ts
{
  data: { mesh: Mesh; peaks: number[] };
  params: ElevationParams;
  constraints: Constraints;
}
```

**Response**
```ts
{ elevationT: Float32Array; elevationR: Float32Array }
```

**Error**
```ts
{ cmd: 'assignElevationError', error: string }
```

Assigns elevation values to triangles and regions using the previously generated mesh.

### `assignRivers`

**Payload**
```ts
{
  data: { mesh: Mesh };
  params: RiverParams;
}
```

**Response**
```ts
{ flowT: Float32Array }
```

**Error**
```ts
{ cmd: 'assignRiversError', error: string }
```

Calculates river flow across the mesh using the elevation data.

### `loadMapJSON`

**Payload**
```ts
string // path to a .map JSON file
```

**Response**
```ts
MapData
```

**Error**
```ts
{ cmd: 'loadMapJSONError', error: string }
```

Loads a map exported from Fantasy Map Generator and returns the parsed JSON data.

### `generateStates`

**Payload**
```ts
{
  cells: Cell[];
  options: StateOptions;
}
```

**Response**
```ts
State[]
```

**Error**
```ts
{ cmd: 'generateStatesError', error: string }
```

Creates state groupings from a list of cells.

### `generateRoads`

**Payload**
```ts
{
  burgs: Burg[];
  params: RoadOptions;
}
```

**Response**
```ts
Road[]
```

**Error**
```ts
{ cmd: 'generateRoadsError', error: string }
```

Generates road connections between burgs according to the given parameters.

## Example

```ts
const worker = new Worker('worker.js');

// Generate mesh
worker.postMessage({ cmd: 'generateMesh', payload: undefined });

worker.onmessage = (event) => {
  const { cmd, data } = event.data;
  if (cmd === 'generateMesh') {
    console.log('Mesh ready', data);
    // Now assign elevation
    worker.postMessage({
      cmd: 'assignElevation',
      payload: {
        data,
        params: { noisyCoastlines: 0.5, hillHeight: 0.3, mountainSharpness: 0.8, oceanDepth: 1.0 },
        constraints: { size: 1024, constraints: new Float32Array() },
      },
    });
  } else if (cmd === 'assignElevation') {
    console.log('Elevation', data);
  }
};
```
