import React from 'react';
import AntFarm from './AntFarm';
import './App.css';

/** Main application component rendering the ant farm simulator. */
const App: React.FC = () => {
  return (
    <div className="App">
      <AntFarm />
    </div>
  );
};

export default App;

