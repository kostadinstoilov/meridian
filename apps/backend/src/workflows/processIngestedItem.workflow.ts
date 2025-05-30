import { WorkflowEntrypoint, type WorkflowEvent, type WorkflowStep, type WorkflowStepConfig } from 'cloudflare:workers';
import {
  $data_sources,
  $ingested_items,
  and,
  eq,
  gte,
  inArray,
  isNull,
  type DataSourceConfigWrapper,
} from '@meridian/database';
import { err, ok } from 'neverthrow';
import { ResultAsync } from 'neverthrow';
import type { Env } from '../index';
import { getArticleWithBrowser, getArticleWithFetch } from '../lib/articleFetchers';
import { createEmbeddings } from '../lib/embeddings';
import { Logger } from '../lib/logger';
import { DomainRateLimiter } from '../lib/rateLimiter';
import { getDb } from '../lib/utils';
import type { z } from 'zod';
import { getArticleRepresentationPrompt } from '../prompts/articleRepresentation.prompt';
import { createGoogleGenerativeAI, google } from '@ai-sdk/google';
import { generateText } from 'ai';

const dbStepConfig: WorkflowStepConfig = {
  retries: { limit: 3, delay: '1 second', backoff: 'linear' },
  timeout: '5 seconds',
};

/**
 * Parameters for the ProcessArticles workflow
 */
export type ProcessArticlesParams = { ingested_item_ids: number[] };

const workflowLogger = new Logger({ workflow: 'ProcessArticles' });

/**
 * Workflow that processes articles by fetching their content, extracting text with Readability,
 * generating embeddings, and storing the results.
 *
 * This workflow handles:
 * - Source type dispatching (RSS, etc.)
 * - Fetching article content with appropriate rate limiting
 * - Domain-specific fetching strategies (browser vs. simple fetch)
 * - Mozilla Readability-based content extraction
 * - 10KB threshold for content storage (DB vs R2)
 * - Embedding generation for search
 * - Persistent storage in database and object storage
 * - Error handling and status tracking
 */
export class ProcessIngestedItemWorkflow extends WorkflowEntrypoint<Env, ProcessArticlesParams> {
  /**
   * Main workflow execution method that processes a batch of articles
   *
   * @param _event Workflow event containing article IDs to process
   * @param step Workflow step context for creating durable operations
   */
  async run(_event: WorkflowEvent<ProcessArticlesParams>, step: WorkflowStep) {
    const env = this.env;
    const db = getDb(env.HYPERDRIVE);
    const google = createGoogleGenerativeAI({
      apiKey: env.GEMINI_API_KEY,
      baseURL: env.GEMINI_BASE_URL || 'https://generativelanguage.googleapis.com/v1beta',
    });
    const logger = workflowLogger.child({
      workflow_id: _event.instanceId,
      initial_article_count: _event.payload.ingested_item_ids.length,
    });

    logger.info('Starting workflow run');

    const articles = await step.do('get articles', dbStepConfig, async () =>
      db
        .select({
          id: $ingested_items.id,
          url: $ingested_items.url_to_original,
          title: $ingested_items.display_title,
          publishedAt: $ingested_items.published_at,
          sourceType: $data_sources.source_type,
          config: $data_sources.config,
        })
        .from($ingested_items)
        .innerJoin($data_sources, eq($ingested_items.data_source_id, $data_sources.id))
        .where(
          and(
            // only process articles that haven't been processed yet
            isNull($ingested_items.processed_at),
            // only process articles that have a publish date in the last 48 hours
            gte($ingested_items.published_at, new Date(new Date().getTime() - 48 * 60 * 60 * 1000)),
            // only articles that have not failed
            isNull($ingested_items.fail_reason),
            // MAIN FILTER: only articles that need to be processed
            inArray($ingested_items.id, _event.payload.ingested_item_ids)
          )
        )
    );

    const fetchLogger = logger.child({ articles_count: articles.length });
    fetchLogger.info('Fetching article contents');

    // Create rate limiter with article processing specific settings
    const rateLimiter = new DomainRateLimiter<{
      id: number;
      url: string;
      title: string | null;
      publishedAt: Date | null;
      sourceType: 'RSS';
      config: z.infer<typeof DataSourceConfigWrapper>;
    }>({ maxConcurrent: 8, globalCooldownMs: 1_000, domainCooldownMs: 5_000 });

    // Process articles with rate limiting and source type dispatcher
    const articlesToProcess: Array<{
      id: number;
      title: string;
      url: string;
      contentBodyText: string;
      contentBodyR2Key: string | null;
      wordCount: number;
      publishedTime?: string;
    }> = [];
    const articleResults = await rateLimiter.processBatch(articles, step, async article => {
      const scrapeLogger = fetchLogger.child({ article_id: article.id, source_type: article.sourceType });

      // Skip PDFs immediately
      if (article.url.toLowerCase().endsWith('.pdf')) {
        scrapeLogger.info('Skipping PDF article');

        // Update the article status to mark it as skipped PDF
        await step.do(`mark PDF article ${article.id} as skipped`, dbStepConfig, async () => {
          return db
            .update($ingested_items)
            .set({
              status: 'SKIPPED_PDF',
              processed_at: new Date(),
              fail_reason: 'PDF article - cannot process',
            })
            .where(eq($ingested_items.id, article.id));
        });

        return { id: article.id, success: false, error: 'pdf' };
      }

      // Dispatcher based on source type
      if (article.sourceType === 'RSS') {
        return await this._processRSSArticle(article, scrapeLogger, step, env);
      }

      scrapeLogger.error('Unsupported source type', { source_type: article.sourceType });
      return { id: article.id, success: false, error: `Unsupported source type: ${article.sourceType}` };
    });

    // Handle results
    let successCount = 0;
    let failCount = 0;

    const dbUpdateLogger = fetchLogger.child({ results_count: articleResults.length });

    for (const result of articleResults) {
      const articleLogger = dbUpdateLogger.child({ article_id: result.id });

      if (result.success && 'processedContent' in result) {
        successCount++;
        articlesToProcess.push({
          id: result.id,
          title: result.processedContent.title,
          url: result.processedContent.url,
          contentBodyText: result.processedContent.contentBodyText,
          contentBodyR2Key: result.processedContent.contentBodyR2Key,
          wordCount: result.processedContent.wordCount,
          publishedTime: result.processedContent.publishedTime,
        });

        await step.do(`update db for successful article ${result.id}`, dbStepConfig, async () => {
          articleLogger.debug('Updating article status to CONTENT_FETCHED');
          return db
            .update($ingested_items)
            .set({
              status: 'PROCESSED',
              usedBrowser: result.used_browser,
            })
            .where(eq($ingested_items.id, result.id));
        });
      } else {
        failCount++;
        // update failed articles in DB with the fail reason
        await step.do(`update db for failed article ${result.id}`, dbStepConfig, async () => {
          const failReason = result.error ? String(result.error) : 'Unknown error';
          const status = result.error?.includes('render') ? 'FAILED_RENDER' : 'FAILED_FETCH';

          articleLogger.warn('Marking article as failed during content fetch', {
            fail_reason: failReason,
            status,
          });

          return db
            .update($ingested_items)
            .set({
              processed_at: new Date(),
              fail_reason: failReason,
              status: status,
            })
            .where(eq($ingested_items.id, result.id));
        });
      }
    }

    const processingLogger = logger.child({
      processing_batch_size: articlesToProcess.length,
      fetch_success_count: successCount,
      fetch_fail_count: failCount,
    });

    processingLogger.info('Processing articles with content extraction and embeddings');

    // process articles for embeddings
    const analysisResults = await Promise.allSettled(
      articlesToProcess.map(async article => {
        const articleLogger = processingLogger.child({ article_id: article.id });
        articleLogger.info('Generating article representation');

        // Analyze article
        const articleRepresentation = await step.do(
          `analyze article ${article.id}`,
          { retries: { limit: 3, delay: '2 seconds', backoff: 'exponential' }, timeout: '1 minute' },
          async () => {
            const response = await generateText({
              model: google('gemini-2.0-flash-001'),
              temperature: 0,
              prompt: getArticleRepresentationPrompt(article.title, article.url, article.contentBodyText),
            });
            return response.text;
          }
        );

        articleLogger.info('Embedding article representation');

        // Generate embeddings (no need to upload to R2 as it's already handled in processing)
        const embeddingResult = await step.do(`generate embeddings for article ${article.id}`, async () => {
          articleLogger.info('Generating embeddings');
          const embeddings = await createEmbeddings(env, [articleRepresentation]);
          if (embeddings.isErr()) throw embeddings.error;
          return embeddings.value[0];
        });

        // handle results in a separate step
        await step.do(`update article ${article.id} status`, async () =>
          db
            .update($ingested_items)
            .set({
              processed_at: new Date(),
              display_title: article.title,
              content_body_text: article.contentBodyText,
              content_body_r2_key: article.contentBodyR2Key,
              embedding: embeddingResult,
              status: 'PROCESSED',
              word_count: article.wordCount,
            })
            .where(eq($ingested_items.id, article.id))
        );

        articleLogger.info('Article processed successfully');

        return { id: article.id, success: true };
      })
    );

    const successfulAnalyses = analysisResults.filter(
      (result): result is PromiseFulfilledResult<{ id: number; success: true }> =>
        result.status === 'fulfilled' && result.value.success
    ).length;

    const failedAnalyses = analysisResults.filter(
      result => result.status === 'rejected' || (result.status === 'fulfilled' && !result.value.success)
    ).length;

    logger.info('Workflow completed', {
      total_articles: articlesToProcess.length,
      successful_analyses: successfulAnalyses,
      failed_analyses: failedAnalyses,
    });
  }

  /**
   * Processes RSS articles by fetching HTML content and using Readability for extraction
   */
  private async _processRSSArticle(
    article: {
      id: number;
      url: string;
      title: string | null;
      publishedAt: Date | null;
      sourceType: 'RSS';
      config: z.infer<typeof DataSourceConfigWrapper>;
    },
    scrapeLogger: Logger,
    step: WorkflowStep,
    env: Env
  ) {
    scrapeLogger.info('Processing RSS article');

    // This will contain either a successful result or a controlled error
    // biome-ignore lint/suspicious/noImplicitAnyLet: <explanation>
    let result;
    try {
      result = await step.do(
        `scrape RSS article ${article.id}`,
        { retries: { limit: 3, delay: '2 second', backoff: 'exponential' }, timeout: '2 minutes' },
        async () => {
          // During retries, let errors bubble up naturally
          if (article.config.config.rss_paywall === true) {
            scrapeLogger.info('Using browser to fetch article (tricky domain)');
            const browserResult = await getArticleWithBrowser(env, article.url);
            if (browserResult.isErr()) throw browserResult.error.error;

            return {
              id: article.id,
              success: true,
              parsedContent: browserResult.value,
              used_browser: true,
            };
          }

          scrapeLogger.info('Attempting fetch-first approach');
          const fetchResult = await getArticleWithFetch(article.url);
          if (!fetchResult.isErr()) {
            return {
              id: article.id,
              success: true,
              parsedContent: fetchResult.value,
              used_browser: false,
            };
          }

          // Fetch failed, try browser with jitter
          scrapeLogger.info('Fetch failed, falling back to browser');
          const jitterTime = Math.random() * 2500 + 500;
          await step.sleep('jitter', jitterTime);

          const browserResult = await getArticleWithBrowser(env, article.url);
          if (browserResult.isErr()) throw browserResult.error.error;

          return {
            id: article.id,
            success: true,
            parsedContent: browserResult.value,
            used_browser: true,
          };
        }
      );
    } catch (error) {
      scrapeLogger.error(
        'Failed to scrape RSS article',
        { error: error instanceof Error ? error.message : String(error) },
        error instanceof Error ? error : new Error(String(error))
      );
      // After all retries failed, return a structured error
      return {
        id: article.id,
        success: false,
        error: error instanceof Error ? error.message : String(error) || 'exhausted all retries',
      };
    }

    // Apply 10KB threshold logic
    const CONTENT_SIZE_THRESHOLD = 10240; // 10KB
    const fullText = result.parsedContent.text;
    const wordCount = fullText.split(' ').length;

    let contentBodyText: string;
    let contentBodyR2Key: string | null = null;

    if (fullText.length <= CONTENT_SIZE_THRESHOLD) {
      // Store full text in DB
      contentBodyText = fullText;
    } else {
      // Store first 10KB in DB with truncation indicator, full text in R2
      contentBodyText = `${fullText.substring(0, CONTENT_SIZE_THRESHOLD)}...`;

      // Store full content in R2
      const date = result.parsedContent.publishedTime ? new Date(result.parsedContent.publishedTime) : new Date();
      const r2Key = `processed_content/${date.getUTCFullYear()}/${date.getUTCMonth() + 1}/${date.getUTCDate()}/${article.id}.txt`;

      try {
        await env.ARTICLES_BUCKET.put(r2Key, fullText);
        contentBodyR2Key = r2Key;
        scrapeLogger.info('Stored full content in R2', { r2_key: r2Key, content_length: fullText.length });
      } catch (r2Error) {
        scrapeLogger.error('Failed to store content in R2', { r2_key: r2Key }, r2Error as Error);
        // Continue with truncated content in DB only
      }
    }

    return {
      id: article.id,
      success: true,
      processedContent: {
        title: result.parsedContent.title,
        contentBodyText,
        contentBodyR2Key,
        url: article.url,
        wordCount,
        publishedTime: result.parsedContent.publishedTime,
      },
      used_browser: result.used_browser,
    };
  }
}

/**
 * Starts a new ProcessArticles workflow instance with the provided article IDs
 *
 * @param env Application environment
 * @param params Parameters containing the list of article IDs to process
 * @returns Result containing either the created workflow or an error
 */
export async function startProcessArticleWorkflow(env: Env, params: ProcessArticlesParams) {
  const workflow = await ResultAsync.fromPromise(
    env.PROCESS_INGESTED_ITEM.create({ id: crypto.randomUUID(), params }),
    e => (e instanceof Error ? e : new Error(String(e)))
  );
  if (workflow.isErr()) return err(workflow.error);
  return ok(workflow.value);
}
