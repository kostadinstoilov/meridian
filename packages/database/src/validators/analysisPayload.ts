import { z } from 'zod';
import type { DataSourceConfigWrapper, RssSourceConfigV1 } from './dataSourceConfig';

// RSS Source Configuration Schema v1.0
export const AnalysisPayloadBaseV1 = z.object({
  schema_version: z.literal('1.0'),
  analysis_type: z.string(),
  data: z.record(z.unknown()),
});

// Base Analysis Payload Wrapper
// Discriminated union that can wrap different analysis payloads
export const AnalysisPayloadWrapper = z.discriminatedUnion('analysis_type', [
  z.object({
    analysis_type: z.literal('RSS'),
    data: AnalysisPayloadBaseV1,
  }),
]);

// Type exports for TypeScript usage
export type RssSourceConfigV1Type = z.infer<typeof RssSourceConfigV1>;
export type DataSourceConfigWrapperType = z.infer<typeof DataSourceConfigWrapper>;
