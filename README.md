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

## Forex Strategy Research Lab (Python)

This repository now also includes a modular Python research framework under `forex_research_lab/`.

Run the initial prototype:

```bash
pip3 install -r requirements.txt
python3 scripts/run_prototype.py
```

Prototype outputs are written to `outputs/forex_research_prototype/`.
