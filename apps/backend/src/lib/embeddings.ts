import { err, ok } from 'neverthrow';
import { z } from 'zod';
import type { Env } from '../index';
import { tryCatchAsync } from './tryCatchAsync';

const embeddingsResponseSchema = z.object({
  embeddings: z.array(z.array(z.number())),
});

export async function createEmbeddings(env: Env, texts: string[]) {
  const response = await tryCatchAsync(
    fetch(`${env.MERIDIAN_ML_SERVICE_URL}/embeddings`, {
      method: 'POST',
      body: JSON.stringify({ texts }),
      headers: {
        Authorization: `Bearer ${env.MERIDIAN_ML_SERVICE_API_KEY}`,
        'Content-Type': 'application/json',
      },
    })
  );
  if (response.isErr()) {
    return err(response.error);
  }
  if (!response.value.ok) {
    return err(new Error(`Failed to fetch embeddings: ${response.value.statusText}`));
  }

  const jsonResult = await tryCatchAsync(response.value.json());
  if (jsonResult.isErr()) {
    return err(jsonResult.error);
  }

  const parsedResponse = embeddingsResponseSchema.safeParse(jsonResult.value);
  if (parsedResponse.success === false) {
    return err(new Error(`Invalid response ${JSON.stringify(parsedResponse.error)}`));
  }

  return ok(parsedResponse.data.embeddings);
}
