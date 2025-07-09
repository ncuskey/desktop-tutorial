import OpenAI from 'openai';

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

export async function generateAdventureHooksFromLore(lore: any) {
  const prompt = `Given world lore: ${JSON.stringify(lore)} generate 3 new adventure hooks as JSON array.`;
  const chatRes = await openai.chat.completions.create({
    model: 'gpt-4o',
    messages: [{ role: 'user', content: prompt }],
  });
  return JSON.parse(chatRes.choices[0].message.content);
}
