import request from 'supertest';
import app from '../src/server';

jest.mock('../src/api/lore', () => ({
  generateLore: jest.fn().mockResolvedValue({
    title: 'Title',
    type: 'Type',
    tags: ['tag'],
    lore: 'Lore text'
  })
}));

jest.mock('../src/api/hooks', () => ({
  generateAdventureHooksFromLore: jest.fn().mockResolvedValue(['hook1', 'hook2'])
}));

describe('POST /api/lore', () => {
  it('returns 400 when `topic` is missing', async () => {
    const res = await request(app).post('/api/lore').send({});
    expect(res.status).toBe(400);
  });

  it('returns 200 and JSON with required keys when `topic` is provided', async () => {
    const res = await request(app).post('/api/lore').send({ topic: 'castle' });
    expect(res.status).toBe(200);
    expect(res.body).toEqual(
      expect.objectContaining({
        title: expect.any(String),
        type: expect.any(String),
        tags: expect.anything(),
        lore: expect.any(String)
      })
    );
  });
});

describe('POST /api/hooks', () => {
  it('returns 400 when `lore` is missing', async () => {
    const res = await request(app).post('/api/hooks').send({});
    expect(res.status).toBe(400);
  });

  it('returns 200 and JSON array when `lore` is provided', async () => {
    const res = await request(app).post('/api/hooks').send({ lore: {} });
    expect(res.status).toBe(200);
    expect(Array.isArray(res.body)).toBe(true);
  });
});
