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

const API_URL = 'https://api.openai.com/v1/chat/completions';
// TODO: insert API key securely
const API_KEY = '<API_KEY>';

type ChatMessage = { role: 'system' | 'user' | 'assistant'; content: string };

async function callChatAPI(messages: ChatMessage[]) {
  const res = await fetch(API_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${API_KEY}`,
    },
    body: JSON.stringify({ model: 'gpt-3.5-turbo', messages }),
  });
  if (!res.ok) {
    throw new Error(`OpenAI request failed: ${res.status}`);
  }
  return res.json();
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
  const prompt = `Create initial world lore using these states: ${JSON.stringify(states)}. Return JSON.`;
  const data = await callChatAPI([{ role: 'user', content: prompt }]);
  // TODO: handle parse errors
  return JSON.parse(data.choices[0].message.content) as WorldLore;
}

/**
 * Update lore after a quest completes.
 *
 * The prompt references the existing lore and quest outcome and expects
 * updated lore in JSON format.
 */
export async function applyOutcome(outcome: QuestOutcome): Promise<WorldLore> {
  const prompt = `Given the current lore and the party completed quest ${outcome.questID} (${outcome.success}), return updated lore as JSON.`;
  const data = await callChatAPI([{ role: 'user', content: prompt }]);
  return JSON.parse(data.choices[0].message.content) as WorldLore;
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
  const prompt = `Given world lore: ${JSON.stringify(lore)} generate 3 new adventure hooks as JSON array.`;
  const data = await callChatAPI([{ role: 'user', content: prompt }]);
  return JSON.parse(data.choices[0].message.content) as AdventureHook[];
}

