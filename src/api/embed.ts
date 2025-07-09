import OpenAI from 'openai';
import { pinecone } from './pineconeClient';

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

export async function embedAndStoreLoreEntry(
  id: string,
  title: string,
  text: string,
  tags: string[] = []
) {
  const { data } = await openai.embeddings.create({
    model: 'text-embedding-3-small',
    input: text,
  });
  const vector = data[0].embedding;
  const index = pinecone.Index(process.env.PINECONE_INDEX_NAME!);
  await index.upsert([{ id, values: vector, metadata: { title, text, tags } }]);
}
