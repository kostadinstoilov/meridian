import app from './app';
import { SourceScraperDO } from './durable_objects/sourceScraperDO';
import { Logger } from './lib/logger';
import { startProcessArticleWorkflow } from './workflows/processArticles.workflow';

type ArticleQueueMessage = { articles_id: number[] };

export type Env = {
  // Bindings
  ARTICLES_BUCKET: R2Bucket;
  ARTICLE_PROCESSING_QUEUE: Queue<ArticleQueueMessage>;
  SOURCE_SCRAPER: DurableObjectNamespace<SourceScraperDO>;
  PROCESS_ARTICLES: Workflow;
  HYPERDRIVE: Hyperdrive;

  // Secrets
  API_TOKEN: string;

  AXIOM_DATASET: string | undefined; // optional, use if you want to send logs to axiom
  AXIOM_TOKEN: string | undefined; // optional, use if you want to send logs to axiom

  CLOUDFLARE_API_TOKEN: string;
  CLOUDFLARE_ACCOUNT_ID: string;

  DATABASE_URL: string;

  GEMINI_API_KEY: string;
  GEMINI_BASE_URL: string;

  MERIDIAN_ML_SERVICE_URL: string;
  MERIDIAN_ML_SERVICE_API_KEY: string;
};

// Create a base logger for the queue handler
const queueLogger = new Logger({ service: 'article-queue-handler' });

export default {
  fetch: app.fetch,
  async queue(batch: MessageBatch<unknown>, env: Env): Promise<void> {
    const batchLogger = queueLogger.child({ batch_size: batch.messages.length });
    batchLogger.info('Received batch of articles to process');

    const articlesToProcess: number[] = [];
    for (const message of batch.messages) {
      const { articles_id } = message.body as ArticleQueueMessage;
      batchLogger.debug('Processing message', { message_id: message.id, article_count: articles_id.length });

      for (const id of articles_id) {
        articlesToProcess.push(id);
      }
    }

    batchLogger.info('Articles extracted from batch', { total_articles: articlesToProcess.length });

    if (articlesToProcess.length === 0) {
      batchLogger.info('Queue batch was empty, nothing to process');
      batch.ackAll(); // Acknowledge the empty batch
      return;
    }

    // Process articles in chunks of 96
    const CHUNK_SIZE = 96;
    const articleChunks = [];
    for (let i = 0; i < articlesToProcess.length; i += CHUNK_SIZE) {
      articleChunks.push(articlesToProcess.slice(i, i + CHUNK_SIZE));
    }

    batchLogger.info('Split articles into chunks', { chunk_count: articleChunks.length });

    // Process each chunk sequentially
    for (const chunk of articleChunks) {
      const workflowResult = await startProcessArticleWorkflow(env, { articles_id: chunk });
      if (workflowResult.isErr()) {
        batchLogger.error(
          'Failed to trigger ProcessArticles Workflow',
          { error_message: workflowResult.error.message, chunk_size: chunk.length },
          workflowResult.error
        );
        // Retry the entire batch if Workflow creation failed
        batch.retryAll({ delaySeconds: 30 }); // Retry after 30 seconds
        return;
      }

      batchLogger.info('Successfully triggered ProcessArticles Workflow for chunk', {
        workflow_id: workflowResult.value.id,
        chunk_size: chunk.length,
      });
    }

    batch.ackAll(); // Acknowledge the entire batch after all chunks are processed
  },
} satisfies ExportedHandler<Env>;

export { SourceScraperDO };
export { ProcessArticles } from './workflows/processArticles.workflow';
