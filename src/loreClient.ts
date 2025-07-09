import type { State } from '../worker';
import type { AdventureHook } from './hooks/useWorld';

/** Outcome of a quest reported to the lore engine. */
export interface QuestOutcome {
  questID: string;
  success: boolean;
}

/**
 * World lore structure returned by the LoreEngine.
 * Additional fields may be included depending on prompts.
 */
export interface WorldLore {
  summary: string;
  // TODO: add more descriptive fields as needed
}


/**
 * Initialize lore for a newly generated world.
 *
 * Example payload sent to OpenAI:
 * ```json
 * {
 *   "model": "gpt-3.5-turbo",
 *   "messages": [{ "role": "user", "content": "Create lore for ..." }]
 * }
 * ```
 * The assistant should respond with JSON text matching `WorldLore`.
 */
export async function initializeLore(states: State[]): Promise<WorldLore> {
  const topic = `Create initial world lore using these states: ${JSON.stringify(states)}`;
  const res = await fetch('/api/lore', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ topic }),
  });
  if (!res.ok) {
    throw new Error('Lore API error');
  }
  return res.json() as Promise<WorldLore>;
}

/**
 * Update lore after a quest completes.
 *
 * The prompt references the existing lore and quest outcome and expects
 * updated lore in JSON format.
 */
export async function applyOutcome(outcome: QuestOutcome): Promise<WorldLore> {
  const topic = `Quest ${outcome.questID} ${outcome.success ? 'succeeded' : 'failed'}`;
  const res = await fetch('/api/lore', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ topic }),
  });
  if (!res.ok) {
    throw new Error('Lore API error');
  }
  return res.json() as Promise<WorldLore>;
}

/**
 * Generate a list of adventure hooks from world lore.
 *
 * Example assistant response:
 * ```json
 * [{ "id": "1", "description": "Rescue the merchant" }]
 * ```
 */
export async function generateAdventureHooks(lore: WorldLore): Promise<AdventureHook[]> {
  const res = await fetch('/api/hooks', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ lore }),
  });
  if (!res.ok) {
    throw new Error('Hooks API error');
  }
  return res.json() as Promise<AdventureHook[]>;
}

