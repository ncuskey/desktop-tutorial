# MapEngine Web App

This project uses React and TypeScript to render and manipulate fantasy maps in the browser. A WebWorker (`src/worker.ts`) performs heavy map generation tasks so the UI stays responsive.

## Getting Started

```bash
npm install
npm start      # launches CRA dev server
npm run build  # creates production build
npm test       # runs unit tests
```

## Public Assets

The sample map JSON is served from /maps/sample.map.json in the public folder.
