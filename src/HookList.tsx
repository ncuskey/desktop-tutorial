import React from 'react';
import type { AdventureHook } from './hooks/useWorld';

interface Props {
  hooks: AdventureHook[];
  completeHook(id: string, success: boolean): void;
}

/**
 * Simple list of adventure hooks with completion buttons.
 */
const HookList: React.FC<Props> = ({ hooks, completeHook }) => {
  return (
    <ul>
      {hooks.map((hook) => (
        <li key={hook.id}>
          {hook.description}
          {hook.completed ? (
            <span> - completed</span>
          ) : (
            <button onClick={() => completeHook(hook.id, true)}>Complete</button>
          )}
        </li>
      ))}
    </ul>
  );
};

export default HookList;
