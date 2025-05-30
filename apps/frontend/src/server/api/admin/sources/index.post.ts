import { getDB } from '~/server/lib/utils';
import { z } from 'zod';
import type { DataSourceConfigWrapper } from '@meridian/database';
import { $data_sources } from '@meridian/database';

const schema = z.object({
  url: z.string().url(),
});

export default defineEventHandler(async event => {
  await requireUserSession(event); // require auth

  const bodyResult = schema.safeParse(await readBody(event));
  if (bodyResult.success === false) {
    throw createError({ statusCode: 400, statusMessage: 'Invalid request body' });
  }

  try {
    await getDB(event)
      .insert($data_sources)
      .values({
        name: 'Unknown',
        source_type: 'RSS',
        config: {
          source_type: 'RSS',
          config: {
            config_schema_version: '1.0',
            rss_paywall: false,
            url: bodyResult.data.url,
          },
        } satisfies z.infer<typeof DataSourceConfigWrapper>,
      });
  } catch (error) {
    console.error('Failed to add source', error);
    throw createError({ statusCode: 500, statusMessage: 'Failed to add source' });
  }

  const config = useRuntimeConfig();

  try {
    await fetch(`${config.public.WORKER_API}/do/admin/initialize-dos`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${config.worker.api_token}`,
      },
    });
  } catch (error) {
    console.error('Failed to initialize DOs', error);
    throw createError({ statusCode: 500, statusMessage: 'Failed to initialize DOs' });
  }

  return {
    success: true,
  };
});
