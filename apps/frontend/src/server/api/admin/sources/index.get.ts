import { sql, $ingested_items, and, gte } from '@meridian/database';
import { getDB } from '~/server/lib/utils';

export default defineEventHandler(async event => {
  await requireUserSession(event); // require auth

  const db = getDB(event);
  const sources = await db.query.$data_sources.findMany();
  if (sources.length === 0) {
    return { overview: null, sources: [] };
  }

  // get article stats for last 7 days
  const sevenDaysAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000);
  const articleStats = await db.query.$ingested_items.findMany({
    where: sql`ingested_at >= ${sevenDaysAgo.toISOString()}`,
    columns: {
      data_source_id: true,
      status: true,
      ingested_at: true,
      processed_at: true,
    },
  });

  // calculate per-source stats
  const sourceStats = sources.map(source => {
    const sourceArticles = articleStats.filter(a => a.data_source_id === source.id);
    const last24hArticles = sourceArticles.filter(
      a => a.ingested_at && new Date(a.ingested_at) > new Date(Date.now() - 24 * 60 * 60 * 1000)
    );

    // calculate health metrics
    const totalArticles = sourceArticles.length;
    const processedArticles = sourceArticles.filter(a => a.status === 'PROCESSED');
    const failedArticles = sourceArticles.filter(a => a.status?.endsWith('_FAILED'));

    // calculate processing time for processed articles
    const processingTimes = processedArticles
      .map(a =>
        a.processed_at && a.ingested_at ? new Date(a.processed_at).getTime() - new Date(a.ingested_at).getTime() : null
      )
      .filter(time => time !== null);

    const avgProcessingTime = processingTimes.length
      ? Math.round(processingTimes.reduce((a, b) => a + b, 0) / processingTimes.length / 1000) // in seconds
      : null;

    return {
      id: source.id,
      name: source.name,
      url: source.config.config.url,
      paywall: source.config.config.rss_paywall,
      frequency:
        source.scrape_frequency_minutes <= 60
          ? 'Hourly'
          : source.scrape_frequency_minutes <= 120
            ? '4 Hours'
            : source.scrape_frequency_minutes <= 180
              ? '6 Hours'
              : 'Daily',
      lastChecked: source.lastChecked?.toISOString(),

      // article counts
      totalArticles: sourceArticles.length,
      avgPerDay: last24hArticles.length / 24,

      // health metrics
      processSuccessRate: totalArticles ? (processedArticles.length / totalArticles) * 100 : null,
      errorRate: totalArticles ? (failedArticles.length / totalArticles) * 100 : null,
      avgProcessingTime,
    };
  });

  // get global stats
  const startOfToday = new Date();
  startOfToday.setUTCHours(0, 0, 0, 0);

  const [lastSourceCheck, lastArticleProcessed, lastArticleFetched, todayStats, staleSources] = await Promise.all([
    // get latest source check
    db.query.$data_sources.findFirst({
      orderBy: sql`last_checked DESC NULLS LAST`,
      columns: { lastChecked: true },
    }),
    // get latest processed article
    db.query.$ingested_items.findFirst({
      where: sql`status = 'PROCESSED'`,
      orderBy: sql`processed_at DESC NULLS LAST`,
      columns: { processed_at: true },
    }),
    // get latest fetched article
    db.query.$ingested_items.findFirst({
      orderBy: sql`ingested_at DESC NULLS LAST`,
      columns: { ingested_at: true },
    }),
    // get today's stats
    db.query.$ingested_items.findMany({
      where: and(gte($ingested_items.ingested_at, startOfToday)),
      columns: {
        status: true,
        ingested_at: true,
        processed_at: true,
      },
    }),
    // get stale sources count
    db.query.$data_sources.findMany({
      where: sql`(
        (scrape_frequency_minutes <= 60 AND last_checked < NOW() - INTERVAL '2 hours') OR
        (scrape_frequency_minutes <= 120 AND last_checked < NOW() - INTERVAL '8 hours') OR
        (scrape_frequency_minutes <= 180 AND last_checked < NOW() - INTERVAL '12 hours') OR
        (scrape_frequency_minutes <= 240 AND last_checked < NOW() - INTERVAL '48 hours')
      )`,
      columns: { id: true },
    }),
  ]);

  const overview = {
    lastSourceCheck: lastSourceCheck?.lastChecked?.toISOString() ?? null,
    lastArticleProcessed: lastArticleProcessed?.processed_at?.toISOString() ?? null,
    lastArticleFetched: lastArticleFetched?.ingested_at?.toISOString() ?? null,
    articlesProcessedToday: todayStats.filter(a => a.status === 'PROCESSED').length,
    articlesFetchedToday: todayStats.length,
    errorsToday: todayStats.filter(a => a.status?.endsWith('_FAILED')).length,
    staleSourcesCount: staleSources.length,
    totalSourcesCount: sources.length,
  };

  return {
    overview,
    sources: sourceStats,
  };
});
