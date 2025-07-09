import 'dotenv/config';
import express from 'express';
import { generateLore } from './api/lore';
import { generateAdventureHooksFromLore } from './api/hooks';

const app = express();
app.use(express.json());

app.post('/api/lore', async (req, res) => {
  const { topic } = req.body as { topic?: string };
  if (!topic) {
    res.status(400).json({ error: 'Missing topic' });
    return;
  }
  try {
    const lore = await generateLore(topic);
    res.json(lore);
  } catch (err: any) {
    console.error(err);
    res.status(500).json({ error: err.message ?? 'Error' });
  }
});

app.post('/api/hooks', async (req, res) => {
  const { lore } = req.body as { lore?: any };
  if (!lore) {
    res.status(400).json({ error: 'Missing lore' });
    return;
  }
  try {
    const hooks = await generateAdventureHooksFromLore(lore);
    res.json(hooks);
  } catch (err: any) {
    console.error(err);
    res.status(500).json({ error: err.message ?? 'Error' });
  }
});

export default app;

if (require.main === module) {
  const PORT = process.env.PORT || 3001;
  app.listen(PORT, () => console.log(`Server listening on ${PORT}`));
}
