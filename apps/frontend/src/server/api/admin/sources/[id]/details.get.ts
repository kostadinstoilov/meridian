import { $ingested_items, $data_sources, eq, and, desc, ingestedItemStatusEnum } from '@meridian/database';
import { getDB } from '~/server/lib/utils';

// to access the enums
type ArticleStatus = (typeof ingestedItemStatusEnum.enumValues)[number];

export default defineEventHandler(async event => {
  await requireUserSession(event); // require auth

  const sourceId = Number(getRouterParam(event, 'id'));
  if (Number.isNaN(sourceId)) {
    throw createError({ statusCode: 400, statusMessage: 'Invalid source ID' });
  }

  // get source details
  const db = getDB(event);
  const source = await db.query.$data_sources.findFirst({ where: eq($data_sources.id, sourceId) });
  if (source === undefined) {
    throw createError({ statusCode: 404, statusMessage: 'Source not found' });
  }

  // get query params for filtering and sorting
  const query = getQuery(event);
  const page = Number(query.page) || 1;
  const pageSize = 50;
  const status = query.status as string;
  const sortBy = (query.sortBy as string) || 'createdAt';
  const sortOrder = query.sortOrder === 'asc' ? 'asc' : 'desc';

  // build where clause
  const conditions = [eq($ingested_items.data_source_id, sourceId)];

  // only add conditions if they're valid enum values
  if (ingestedItemStatusEnum.enumValues.includes(status as ArticleStatus)) {
    conditions.push(eq($ingested_items.status, status as ArticleStatus));
  }

  const whereClause = and(...conditions);

  // determine sort field
  const sortField =
    sortBy === 'publishedAt'
      ? $ingested_items.published_at
      : sortBy === 'processedAt'
        ? $ingested_items.processed_at
        : $ingested_items.ingested_at;

  // get articles with filters and sorting
  const articles = await db.query.$ingested_items.findMany({
    where: whereClause,
    orderBy: sortOrder === 'asc' ? sortField : desc(sortField),
    limit: pageSize,
    offset: (page - 1) * pageSize,
  });

  // get total count with filters
  const totalCount = await db.query.$ingested_items.findMany({
    where: whereClause,
    columns: { id: true },
  });

  return {
    id: source.id,
    name: source.name,
    url: source.config.config.url,
    initialized: source.do_initialized_at !== null,
    frequency:
      source.scrape_frequency_minutes <= 60
        ? 'Hourly'
        : source.scrape_frequency_minutes <= 120
          ? '4 Hours'
          : source.scrape_frequency_minutes <= 180
            ? '6 Hours'
            : 'Daily',
    lastFetched: source.lastChecked?.toISOString(),
    articles: articles.map(article => ({
      id: article.id,
      title: article.display_title ?? 'Unknown',
      url: article.url_to_original ?? 'Unknown',
      publishedAt: article.published_at?.toISOString(),
      status: article.status,
      failReason: article.fail_reason,
      processedAt: article.processed_at?.toISOString(),
      createdAt: article.ingested_at?.toISOString(),
      hasEmbedding: article.embedding !== null,
    })),
    pagination: {
      currentPage: page,
      totalPages: Math.ceil(totalCount.length / pageSize),
      totalItems: totalCount.length,
    },
  };
});
