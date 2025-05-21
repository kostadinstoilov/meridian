import { getDb as getDbFromDatabase } from '@meridian/database';
import type { Context } from 'hono';
import { z } from 'zod';
import type { HonoEnv } from '../app';

export function getDb(hyperdrive: Hyperdrive) {
  return getDbFromDatabase(hyperdrive.connectionString, {
    // Workers limit the number of concurrent external connections, so be sure to limit
    // the size of the local connection pool that postgres.js may establish.
    max: 5,

    // If you are not using array types in your Postgres schema,
    // disabling this will save you an extra round-trip every time you connect.
    fetch_types: false,
  });
}

export function hasValidAuthToken(c: Context<HonoEnv>) {
  const auth = c.req.header('Authorization');
  if (auth === undefined || auth !== `Bearer ${c.env.API_TOKEN}`) {
    return false;
  }
  return true;
}

export const userAgents = [
  // ios (golden standard for publishers)
  'Mozilla/5.0 (iPhone; CPU iPhone OS 17_4_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1', // iphone safari (best overall)
  'Mozilla/5.0 (iPhone; CPU iPhone OS 17_4_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/123.0.6312.87 Mobile/15E148 Safari/604.1', // iphone chrome

  // android (good alternatives)
  'Mozilla/5.0 (Linux; Android 14; SM-S908B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Mobile Safari/537.36', // samsung flagship
  'Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Mobile Safari/537.36', // pixel
];
