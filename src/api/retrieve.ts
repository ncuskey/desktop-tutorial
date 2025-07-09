import OpenAI from 'openai';
import { pinecone } from './pineconeClient';

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

export async function retrieveRelevantLore(
  queryText: string,
  topK = 3
): Promise<string[]> {
  const embedRes = await openai.embeddings.create({
    model: 'text-embedding-3-small',
    input: queryText,
  });
  const vector = embedRes.data[0].embedding;
  const index = pinecone.Index(process.env.PINECONE_INDEX_NAME!);
  const queryRes = await index.query({ vector, topK, includeMetadata: true });
  return queryRes.matches?.map(m => (m.metadata as any).text) ?? [];
}
