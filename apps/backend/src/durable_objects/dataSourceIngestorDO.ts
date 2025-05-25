import { DurableObject } from 'cloudflare:workers';
import { $data_sources, $ingested_items, eq } from '@meridian/database';
import { type Result, ResultAsync, err, ok } from 'neverthrow';
import { z } from 'zod';
import type { Env } from '../index';
import { Logger } from '../lib/logger';
import { parseRSSFeed } from '../lib/parsers';
import { tryCatchAsync } from '../lib/tryCatchAsync';
import { getDb } from '../lib/utils';
import { userAgents } from '../lib/utils';

// Import RSS config schema
const RssSourceConfigV1 = z.object({
  url: z.string().url(),
  rss_paywall: z.boolean().optional().default(false),
  config_schema_version: z.literal('1.0'),
});

/**
 * Schema for validating DataSourceState
 * Used to ensure state hasn't been corrupted before operating on it
 */
const DataSourceStateSchema = z.object({
  dataSourceId: z.number().int().positive(),
  sourceType: z.string(),
  config: RssSourceConfigV1, // Assuming RSS for now
  configVersionHash: z.string().nullable(),
  scrapeFrequencyTier: z.number().int().positive(),
  lastChecked: z.number().nullable(),
});

/**
 * State interface for managing data source scraping configuration and status
 */
type DataSourceState = z.infer<typeof DataSourceStateSchema>;

const tierIntervals = {
  1: 60 * 60 * 1000, // Tier 1: Check every hour
  2: 4 * 60 * 60 * 1000, // Tier 2: Check every 4 hours
  3: 6 * 60 * 60 * 1000, // Tier 3: Check every 6 hours
  4: 24 * 60 * 60 * 1000, // Tier 4: Check every 24 hours
};
const DEFAULT_INTERVAL = tierIntervals[2]; // Default to 4 hours if tier is invalid

// --- Retry Configuration ---
const MAX_STEP_RETRIES = 3; // Max retries for *each* step (fetch, parse, insert)
const INITIAL_RETRY_DELAY_MS = 500; // Start delay, doubles each time

/**
 * Executes an operation with exponential backoff retries
 *
 * @param operation Function that returns a Promise<Result> to execute with retries
 * @param maxRetries Maximum number of retry attempts
 * @param initialDelayMs Initial delay between retries in milliseconds (doubles each retry)
 * @param logger Logger instance to record retry attempts and failures
 * @returns Result object from either a successful operation or the last failed attempt
 *
 * @template T Success value type
 * @template E Error type, must extend Error
 */
async function attemptWithRetries<T, E extends Error>(
  operation: () => Promise<Result<T, E>>,
  maxRetries: number,
  initialDelayMs: number,
  logger: Logger
): Promise<Result<T, E>> {
  let lastError: E | undefined;

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    logger.debug(`Attempt ${attempt}/${maxRetries}...`);
    const result = await operation();

    if (result.isOk()) {
      logger.debug(`Attempt ${attempt} successful.`);
      return ok(result.value); // Return successful result immediately
    }

    lastError = result.error; // Store the error
    logger.warn(
      `Attempt ${attempt} failed.`,
      { error_name: lastError.name, error_message: lastError.message },
      lastError
    );

    // If not the last attempt, wait before retrying
    if (attempt < maxRetries) {
      const delay = initialDelayMs * 2 ** (attempt - 1);
      logger.debug('Waiting before next attempt.', { delay_ms: delay });
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }

  // If loop finishes, all retries failed
  logger.error('Failed after max attempts.', { max_retries: maxRetries }, lastError);
  // biome-ignore lint/style/noNonNullAssertion: <explanation>
  return err(lastError!);
}

/**
 * Durable Object for periodically scraping RSS feeds from various sources
 *
 * This DO handles:
 * - Scheduled scraping of RSS sources based on frequency tiers
 * - Fetching and parsing RSS content
 * - Extracting and storing new articles
 * - Sending new articles to a processing queue
 * - Managing state across executions
 * - Handling failures with retries
 */
export class DataSourceIngestorDO extends DurableObject<Env> {
  private logger: Logger;

  /**
   * Initializes the DO with logging
   *
   * @param ctx Durable Object state context
   * @param env Application environment
   */
  constructor(ctx: DurableObjectState, env: Env) {
    super(ctx, env);
    this.logger = new Logger({ durable_object: 'SourceScraperDO', do_id: this.ctx.id.toString() });
    this.logger.info('DO initialized');
  }

  /**
   * Initializes the scraper with data source data and sets up the initial alarm
   *
   * @param dataSourceData Data source configuration including ID, source type, config, config version hash, and scrape frequency tier
   * @throws Error if initialization fails
   */
  async initialize(dataSourceData: {
    id: number;
    source_type: string;
    config: unknown;
    config_version_hash: string | null;
    scrape_frequency_tier: number;
  }): Promise<void> {
    const logger = this.logger.child({
      operation: 'initialize',
      dataSourceId: dataSourceData.id,
      source_type: dataSourceData.source_type,
    });
    logger.info('Initializing with data', { data_source_data: dataSourceData });

    const sourceExistsResult = await ResultAsync.fromPromise(
      getDb(this.env.HYPERDRIVE).query.$data_sources.findFirst({ where: (s, { eq }) => eq(s.id, dataSourceData.id) }),
      e => new Error(`Database query failed: ${e}`)
    );
    if (sourceExistsResult.isErr()) {
      logger.error('Failed to query DB for data source', undefined, sourceExistsResult.error);
      throw sourceExistsResult.error; // Rethrow DB error
    }
    if (!sourceExistsResult.value) {
      logger.warn(
        "Data source doesn't exist in DB. This is likely due to a race condition where the source was deleted after being queued for initialization."
      );
      // Instead of throwing, we'll just return without setting up the DO
      return;
    }

    // Parse the config JSONB using the appropriate Zod schema (assuming RSS for now)
    let parsedConfig: z.infer<typeof RssSourceConfigV1>;
    if (dataSourceData.source_type === 'RSS') {
      const configParseResult = RssSourceConfigV1.safeParse(dataSourceData.config);
      if (!configParseResult.success) {
        logger.error('Failed to parse RSS config', { config: dataSourceData.config, error: configParseResult.error });
        throw new Error(`Invalid RSS config: ${configParseResult.error.message}`);
      }
      parsedConfig = configParseResult.data;
    } else {
      throw new Error(`Unsupported source type: ${dataSourceData.source_type}`);
    }

    const state: DataSourceState = {
      dataSourceId: dataSourceData.id,
      sourceType: dataSourceData.source_type,
      config: parsedConfig,
      configVersionHash: dataSourceData.config_version_hash,
      scrapeFrequencyTier: dataSourceData.scrape_frequency_tier,
      lastChecked: null,
    };

    // Add retry logic for storage operations
    let putSuccess = false;
    for (let i = 0; i < 3 && !putSuccess; i++) {
      try {
        await this.ctx.storage.put('state', state);
        putSuccess = true;
        logger.info('Initialized state successfully.');
      } catch (storageError) {
        logger.warn(`Attempt ${i + 1} to put state failed`, undefined, storageError as Error);
        if (i < 2) await new Promise(res => setTimeout(res, 200 * (i + 1))); // Exponential backoff
      }
    }

    if (!putSuccess) {
      logger.error('Failed to put initial state after retries. DO may be unstable.');
      throw new Error('Failed to persist initial DO state.');
    }

    try {
      // Update the data source's do_initialized_at field
      await getDb(this.env.HYPERDRIVE)
        .update($data_sources)
        .set({ do_initialized_at: new Date() })
        .where(eq($data_sources.id, dataSourceData.id));
    } catch (dbError) {
      logger.error('Failed to update data source do_initialized_at', undefined, dbError as Error);
      throw new Error(
        `Failed to update data source initialization status: ${dbError instanceof Error ? dbError.message : String(dbError)}`
      );
    }

    try {
      // Only set alarm if state was successfully stored
      await this.ctx.storage.setAlarm(Date.now() + 5000);
      logger.info('Initial alarm set.');
    } catch (alarmError) {
      logger.error('Failed to set initial alarm', undefined, alarmError as Error);
      throw new Error(
        `Failed to set initial alarm: ${alarmError instanceof Error ? alarmError.message : String(alarmError)}`
      );
    }
  }

  /**
   * Alarm handler that performs the scheduled data source scraping
   *
   * This method is triggered by the DO alarm and:
   * 1. Retrieves and validates the DataSourceState from storage
   * 2. Checks for config changes by comparing config version hashes
   * 3. Dispatches to the appropriate processing method based on source type
   * 4. Schedules the next alarm
   */
  async alarm(): Promise<void> {
    // Keep logger instance outside try block if possible,
    // but create child logger inside if needed after state is fetched.
    const alarmLogger = this.logger.child({ operation: 'alarm' }); // Initial logger

    try {
      // 1. Retrieve the DataSourceState from storage
      const state = await this.ctx.storage.get<DataSourceState>('state');
      if (state === undefined) {
        this.logger.error('State not found in alarm. Cannot proceed.');
        // Maybe schedule alarm far in the future or log an error to an external system
        // We cannot proceed without state.
        return;
      }

      // Validate state to protect against corruption
      const validatedState = DataSourceStateSchema.safeParse(state);
      if (validatedState.success === false) {
        const logger = this.logger.child({ operation: 'alarm', validation_error: validatedState.error.format() });
        logger.error('State validation failed. Cannot proceed with corrupted state.');
        // Schedule a far-future alarm to prevent continuous failed attempts
        await this.ctx.storage.setAlarm(Date.now() + 24 * 60 * 60 * 1000); // 24 hours
        return;
      }

      let currentState = validatedState.data;
      const { dataSourceId, scrapeFrequencyTier } = currentState;
      const alarmLogger = this.logger.child({ operation: 'alarm', dataSourceId });
      alarmLogger.info('Alarm triggered');

      // 2. Check config hash/timestamp logic
      const configCheckLogger = alarmLogger.child({ step: 'Config Check' });
      const dbConfigResult = await ResultAsync.fromPromise(
        getDb(this.env.HYPERDRIVE).query.$data_sources.findFirst({
          where: (ds, { eq }) => eq(ds.id, dataSourceId),
          columns: { config: true, config_version_hash: true },
        }),
        e => new Error(`Database query failed: ${e}`)
      );

      if (dbConfigResult.isErr()) {
        configCheckLogger.error('Failed to query DB for config check', undefined, dbConfigResult.error);
        return;
      }

      if (!dbConfigResult.value) {
        configCheckLogger.warn('Data source no longer exists in DB. Stopping processing.');
        return;
      }

      const dbConfig = dbConfigResult.value;

      // Check if config has changed
      if (dbConfig.config_version_hash !== currentState.configVersionHash) {
        configCheckLogger.info('Config version hash changed, re-initializing', {
          old_hash: currentState.configVersionHash,
          new_hash: dbConfig.config_version_hash,
        });

        // Parse the new config using the appropriate Zod schema
        let newParsedConfig: z.infer<typeof RssSourceConfigV1>;
        if (currentState.sourceType === 'RSS') {
          const configParseResult = RssSourceConfigV1.safeParse(dbConfig.config);
          if (!configParseResult.success) {
            configCheckLogger.error('Failed to parse new RSS config', {
              config: dbConfig.config,
              error: configParseResult.error,
            });
            return;
          }
          newParsedConfig = configParseResult.data;
        } else {
          configCheckLogger.error('Unsupported source type for config update', { sourceType: currentState.sourceType });
          return;
        }

        // Update internal state with new config
        currentState = {
          ...currentState,
          config: newParsedConfig,
          configVersionHash: dbConfig.config_version_hash,
        };

        // Persist the updated state
        await this.ctx.storage.put('state', currentState);
        configCheckLogger.info('State updated with new config');
      } else {
        configCheckLogger.debug('Config version hash unchanged, proceeding with current config');
      }

      const interval = tierIntervals[scrapeFrequencyTier as keyof typeof tierIntervals] || DEFAULT_INTERVAL;

      // --- Schedule the *next* regular alarm run immediately ---
      // This ensures that even if this current run fails completely after all retries,
      // the process will attempt again later according to its schedule.
      const nextScheduledAlarmTime = Date.now() + interval;
      await this.ctx.storage.setAlarm(nextScheduledAlarmTime);
      alarmLogger.info('Next regular alarm scheduled', { next_alarm: new Date(nextScheduledAlarmTime).toISOString() });

      // 3. Dispatcher logic based on source type
      if (currentState.sourceType === 'RSS') {
        await this._fetchAndProcessRss(currentState, alarmLogger);
      } else {
        alarmLogger.error('Unsupported source type', { sourceType: currentState.sourceType });
        return;
      }
    } catch (error) {
      // Use the latest available logger instance (might be base or detailed)
      const errorLogger = alarmLogger || this.logger;
      errorLogger.error(
        'Unhandled exception occurred within alarm handler',
        { error_name: error instanceof Error ? error.name : 'UnknownError' },
        error instanceof Error ? error : new Error(String(error)) // Log the error object/stack
      );
    }
  }

  /**
   * Private method to fetch and process RSS feeds
   *
   * @param state Current data source state containing RSS config
   * @param logger Logger instance for this operation
   */
  private async _fetchAndProcessRss(state: DataSourceState, logger: Logger): Promise<void> {
    const { dataSourceId, config } = state;

    // --- Workflow Step 1: Fetch Feed with Retries ---
    const fetchLogger = logger.child({ step: 'Fetch' });
    const fetchResult = await attemptWithRetries(
      async () => {
        const respResult = await tryCatchAsync(
          fetch(config.url, {
            method: 'GET',
            headers: {
              'User-Agent': userAgents[Math.floor(Math.random() * userAgents.length)],
              Referer: 'https://www.google.com/',
            },
          })
        );
        if (respResult.isErr()) return err(respResult.error as Error);
        // Ensure response is OK before trying to read body
        if (respResult.value.ok === false) {
          return err(new Error(`Fetch failed with status: ${respResult.value.status} ${respResult.value.statusText}`));
        }
        // Read body - this can also fail
        const textResult = await tryCatchAsync(respResult.value.text());
        if (textResult.isErr()) return err(textResult.error as Error);
        return ok(textResult.value);
      },
      MAX_STEP_RETRIES,
      INITIAL_RETRY_DELAY_MS,
      fetchLogger
    );
    if (fetchResult.isErr()) {
      // Error already logged by attemptWithRetries
      return;
    }
    const feedText = fetchResult.value;

    // --- Workflow Step 2: Parse Feed with Retries ---
    const parseLogger = logger.child({ step: 'Parse' });
    const parseResult = await attemptWithRetries(
      async () => parseRSSFeed(feedText),
      MAX_STEP_RETRIES,
      INITIAL_RETRY_DELAY_MS,
      parseLogger
    );
    if (parseResult.isErr()) {
      // Error already logged by attemptWithRetries
      return;
    }
    const articles = parseResult.value; // Type: ParsedArticle[]

    // --- Process Articles and Store Raw Data in R2 ---
    const now = Date.now();
    const ageThreshold = now - 48 * 60 * 60 * 1000; // 48 hours ago

    const articlesToInsert: Omit<typeof $ingested_items.$inferInsert, 'id'>[] = [];
    const r2Logger = logger.child({ step: 'R2 Storage' });

    for (const article of articles) {
      // 1a. Store raw RSS item object in R2
      // Note: parseRSSFeed returns items with id, title, link, pubDate
      // The parseRSSFeed function actually returns items with an 'id' field, but the type doesn't reflect this
      const itemIdFromSource =
        (article as { id?: string; title: string; link: string; pubDate: Date | null }).id || article.link;
      const r2Key = `raw_items/${dataSourceId}/${itemIdFromSource}_${now}.json`;

      try {
        await this.env.ARTICLES_BUCKET.put(r2Key, JSON.stringify(article), {
          httpMetadata: {
            contentType: 'application/json',
          },
        });
        r2Logger.debug('Stored raw RSS item in R2', { r2_key: r2Key, item_id: itemIdFromSource });
      } catch (r2Error) {
        r2Logger.error(
          'Failed to store raw RSS item in R2',
          { r2_key: r2Key, item_id: itemIdFromSource },
          r2Error as Error
        );
        // Continue processing even if R2 storage fails
      }

      // 2. Prepare data for insertion into $ingested_items
      const publishTimestamp = article.pubDate ? article.pubDate.getTime() : 0;

      articlesToInsert.push({
        data_source_id: dataSourceId,
        item_id_from_source: itemIdFromSource,
        raw_data_r2_key: r2Key,
        url_to_original: article.link,
        published_at: article.pubDate,
        display_title: article.title,
        status: 'NEW',
        ingested_at: new Date(now),
        // Other fields like content_body_text, content_body_r2_key, embedding, analysis_payload are NULL/undefined
      });
    }

    if (articlesToInsert.length === 0) {
      logger.info('No articles found in feed');

      // Successfully processed, update lastChecked
      const updatedState = { ...state, lastChecked: now };
      await this.ctx.storage.put('state', updatedState);
      logger.info('Updated lastChecked', { timestamp: new Date(now).toISOString() });

      // Update data source lastChecked in database with retries
      const sourceUpdateLogger = logger.child({ step: 'Source Update' });
      const sourceUpdateResult = await attemptWithRetries(
        async () =>
          ResultAsync.fromPromise(
            getDb(this.env.HYPERDRIVE)
              .update($data_sources)
              .set({ lastChecked: new Date(now) })
              .where(eq($data_sources.id, dataSourceId)),
            e => (e instanceof Error ? e : new Error(`Source update failed: ${String(e)}`))
          ),
        MAX_STEP_RETRIES,
        INITIAL_RETRY_DELAY_MS,
        sourceUpdateLogger
      );

      if (sourceUpdateResult.isErr()) {
        sourceUpdateLogger.error('Failed to update data source lastChecked after all retries');
        return;
      }
      sourceUpdateLogger.info('Updated data source lastChecked in database');
      return;
    }

    logger.info('Processed articles from feed', {
      total_articles: articlesToInsert.length,
    });

    // --- 3. Batch insert into $ingested_items with conflict handling ---
    const dbLogger = logger.child({ step: 'DB Insert' });
    const insertResult = await attemptWithRetries(
      async () =>
        ResultAsync.fromPromise(
          getDb(this.env.HYPERDRIVE)
            .insert($ingested_items)
            .values(articlesToInsert)
            .onConflictDoNothing({ target: [$ingested_items.data_source_id, $ingested_items.item_id_from_source] })
            .returning({ insertedId: $ingested_items.id, itemId: $ingested_items.item_id_from_source }),
          e => (e instanceof Error ? e : new Error(`DB Insert failed: ${String(e)}`)) // Error mapper
        ),
      MAX_STEP_RETRIES,
      INITIAL_RETRY_DELAY_MS,
      dbLogger
    );
    if (insertResult.isErr()) {
      // Error already logged by attemptWithRetries
      return;
    }

    const insertedRows = insertResult.value; // Type: { insertedId: number, itemId: string }[]
    dbLogger.info('DB Insert completed', { affected_rows: insertedRows.length });

    // 4. Retrieve IDs of newly inserted items (onConflictDoNothing means only new items are returned)
    const newlyInsertedIds = insertedRows.map(row => row.insertedId);

    // --- 5. Send only newly inserted item IDs to queue ---
    if (newlyInsertedIds.length > 0 && this.env.ARTICLE_PROCESSING_QUEUE) {
      const BATCH_SIZE_LIMIT = 100; // Adjust as needed

      const queueLogger = logger.child({ step: 'Queue', total_ids_to_queue: newlyInsertedIds.length });
      queueLogger.info('Sending newly inserted item IDs to queue');

      for (let i = 0; i < newlyInsertedIds.length; i += BATCH_SIZE_LIMIT) {
        const batch = newlyInsertedIds.slice(i, i + BATCH_SIZE_LIMIT);
        queueLogger.debug('Sending batch to queue', { batch_size: batch.length, batch_index: i / BATCH_SIZE_LIMIT });

        this.ctx.waitUntil(
          this.env.ARTICLE_PROCESSING_QUEUE.send({ ingested_item_ids: batch }).catch(queueError => {
            queueLogger.error(
              'Failed to send batch to queue',
              { batch_index: i / BATCH_SIZE_LIMIT, batch_size: batch.length },
              queueError instanceof Error ? queueError : new Error(String(queueError))
            );
          })
        );
      }
    }

    // --- Final Step: Update lastChecked only on full success ---
    logger.info('All steps successful. Updating lastChecked');
    const updatedState = { ...state, lastChecked: now };
    await this.ctx.storage.put('state', updatedState);
    logger.info('Updated lastChecked', { timestamp: new Date(now).toISOString() });

    // Update data source lastChecked in database with retries
    const sourceUpdateLogger = logger.child({ step: 'Source Update' });
    const sourceUpdateResult = await attemptWithRetries(
      async () =>
        ResultAsync.fromPromise(
          getDb(this.env.HYPERDRIVE)
            .update($data_sources)
            .set({ lastChecked: new Date(now) })
            .where(eq($data_sources.id, dataSourceId)),
          e => (e instanceof Error ? e : new Error(`Source update failed: ${String(e)}`))
        ),
      MAX_STEP_RETRIES,
      INITIAL_RETRY_DELAY_MS,
      sourceUpdateLogger
    );

    if (sourceUpdateResult.isErr()) {
      sourceUpdateLogger.error('Failed to update data source lastChecked after all retries');
      return;
    }
    sourceUpdateLogger.info('Updated data source lastChecked in database');
  }

  /**
   * Handles HTTP requests to manage the scraper
   *
   * Supports endpoints:
   * - /trigger: Manually triggers an immediate scrape
   * - /status: Returns the current state and next alarm time
   * - /delete: Deletes the DO
   * - /initialize: Sets up the scraper with a new source configuration
   *
   * @param request The incoming HTTP request
   * @returns HTTP response with appropriate status and data
   */
  async fetch(request: Request): Promise<Response> {
    const url = new URL(request.url);
    const fetchLogger = this.logger.child({ operation: 'fetch', method: request.method, path: url.pathname });
    fetchLogger.info('Received fetch request');

    if (url.pathname === '/trigger') {
      fetchLogger.info('Manual trigger received');
      await this.ctx.storage.setAlarm(Date.now()); // Trigger alarm soon
      return new Response('Alarm set');
    }

    if (url.pathname === '/status') {
      fetchLogger.info('Status request received');
      const state = await this.ctx.storage.get('state');
      const alarm = await this.ctx.storage.getAlarm();
      return Response.json({
        state: state || { error: 'State not initialized' },
        nextAlarmTimestamp: alarm,
      });
    }

    if (url.pathname === '/delete' && request.method === 'DELETE') {
      fetchLogger.info('Delete request received');
      try {
        await this.destroy();
        fetchLogger.info('DO successfully destroyed');
        return new Response('Deleted', { status: 200 });
      } catch (error) {
        fetchLogger.error('Failed to destroy DO', undefined, error instanceof Error ? error : new Error(String(error)));
        return new Response(`Failed to delete: ${error instanceof Error ? error.message : String(error)}`, {
          status: 500,
        });
      }
    } else if (url.pathname === '/initialize' && request.method === 'POST') {
      fetchLogger.info('Initialize request received');
      const dataSourceDataResult = await tryCatchAsync(
        request.json<{
          id: number;
          source_type: string;
          config: unknown;
          config_version_hash: string | null;
          scrape_frequency_tier: number;
        }>()
      );
      if (dataSourceDataResult.isErr()) {
        const error =
          dataSourceDataResult.error instanceof Error
            ? dataSourceDataResult.error
            : new Error(String(dataSourceDataResult.error));

        fetchLogger.error('Initialization failed via fetch', undefined, error);
        return new Response(`Initialization failed: ${error.message}`, { status: 500 });
      }

      const dataSourceData = dataSourceDataResult.value;
      if (
        !dataSourceData ||
        typeof dataSourceData.id !== 'number' ||
        typeof dataSourceData.source_type !== 'string' ||
        typeof dataSourceData.scrape_frequency_tier !== 'number'
      ) {
        fetchLogger.warn('Invalid data source data format received', { received_data: dataSourceData });
        return new Response('Invalid data source data format', { status: 400 });
      }

      try {
        await this.initialize(dataSourceData);
        fetchLogger.info('Initialization successful via API');
        return new Response('Initialized');
      } catch (error) {
        fetchLogger.error(
          'Initialization failed',
          undefined,
          error instanceof Error ? error : new Error(String(error))
        );
        return new Response(`Initialization failed: ${error instanceof Error ? error.message : String(error)}`, {
          status: 500,
        });
      }
    }

    fetchLogger.warn('Path not found');
    return new Response('Not found', { status: 404 });
  }

  /**
   * Cleanup method called when the DO is about to be destroyed
   * Removes all stored state
   */
  async destroy() {
    this.logger.info('Destroy called, deleting storage');
    const state = await this.ctx.storage.get<DataSourceState>('state');
    if (state?.dataSourceId) {
      // Clear the do_initialized_at field when DO is destroyed
      await getDb(this.env.HYPERDRIVE)
        .update($data_sources)
        .set({ do_initialized_at: null })
        .where(eq($data_sources.id, state.dataSourceId));
    }
    await this.ctx.storage.deleteAll();
    this.logger.info('Storage deleted');
  }
}
