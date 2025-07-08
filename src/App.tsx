import React from 'react';
import { useWorld } from './hooks/useWorld';
import MapCanvas from './MapCanvas';
import HookList from './HookList';
import './App.css';

/** Main application component. */
const App: React.FC = () => {
  const { world, status, error, completeHook } = useWorld();

  if (status === 'loading') {
    return <div>Loading…</div>;
  }

  if (status === 'error') {
    return <div className="Error">Error: {error}</div>;
  }

  return (
    <div className="App">
      <div className="MapPane">
        <MapCanvas mesh={world.mesh} states={world.states} />
      </div>
      <div className="HookPane">
        <HookList hooks={world.hooks} completeHook={completeHook} />
      </div>
    </div>
  );
};

export default App;
