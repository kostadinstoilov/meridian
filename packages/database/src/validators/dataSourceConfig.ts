import { z } from 'zod';

// RSS Source Configuration Schema v1.0
export const RssSourceConfigV1 = z.object({
  url: z.string().url(),
  rss_paywall: z.boolean().optional().default(false),
  config_schema_version: z.literal('1.0'),
});

// Base Data Source Configuration Wrapper
// Discriminated union that can wrap different source configs
export const DataSourceConfigWrapper = z.discriminatedUnion('source_type', [
  z.object({
    source_type: z.literal('RSS'),
    config: RssSourceConfigV1,
  }),
]);

// Type exports for TypeScript usage
export type RssSourceConfigV1Type = z.infer<typeof RssSourceConfigV1>;
export type DataSourceConfigWrapperType = z.infer<typeof DataSourceConfigWrapper>;
