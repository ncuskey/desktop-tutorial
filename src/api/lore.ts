import { retrieveRelevantLore } from './retrieve';
import { embedAndStoreLoreEntry } from './embed';
import OpenAI from 'openai';

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

export async function generateLore(userTopic: string) {
  const contextSnippets = await retrieveRelevantLore(userTopic);
  const prompt = `
You are a master world-builder AI. Use the context below to write consistent lore.

CONTEXT:
${contextSnippets.join('\n---\n')}

TASK: Write lore about "${userTopic}". Return JSON with { title, type, tags, lore }.
`;
  const chatRes = await openai.chat.completions.create({
    model: 'gpt-4o',
    messages: [{ role: 'system', content: prompt }],
  });
  const newLore = JSON.parse(chatRes.choices[0].message.content);
  await embedAndStoreLoreEntry(
    `lore-${Date.now()}`,
    newLore.title,
    newLore.lore,
    newLore.tags
  );
  return newLore;
}
