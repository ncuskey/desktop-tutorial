import React from 'react';
import { useWorld } from './hooks/useWorld';
import MapCanvas from './MapCanvas';
import HookList from './HookList';

/** Main application component. */
const App: React.FC = () => {
  const { world, completeHook } = useWorld();

  return (
    <div className="App">
      <MapCanvas mesh={world.mesh} states={world.states} />
      <HookList hooks={world.hooks} completeHook={completeHook} />
    </div>
  );
};

export default App;
