CREATE TYPE "public"."article_status" AS ENUM('PENDING_FETCH', 'CONTENT_FETCHED', 'PROCESSED', 'SKIPPED_PDF', 'FETCH_FAILED', 'RENDER_FAILED', 'EMBEDDING_FAILED', 'R2_UPLOAD_FAILED', 'SKIPPED_TOO_OLD');--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "articles" (
	"id" bigserial PRIMARY KEY NOT NULL,
	"title" text NOT NULL,
	"url" text NOT NULL,
	"publish_date" timestamp,
	"status" "article_status" DEFAULT 'PENDING_FETCH',
	"content_file_key" text,
	"word_count" integer,
	"used_browser" boolean,
	"embedding" vector(384),
	"fail_reason" text,
	"source_id" integer NOT NULL,
	"processed_at" timestamp,
	"created_at" timestamp DEFAULT CURRENT_TIMESTAMP,
	CONSTRAINT "articles_url_unique" UNIQUE("url")
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "newsletter" (
	"id" serial PRIMARY KEY NOT NULL,
	"email" text NOT NULL,
	"created_at" timestamp DEFAULT CURRENT_TIMESTAMP,
	CONSTRAINT "newsletter_email_unique" UNIQUE("email")
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "reports" (
	"id" serial PRIMARY KEY NOT NULL,
	"title" text NOT NULL,
	"content" text NOT NULL,
	"total_articles" integer NOT NULL,
	"total_sources" integer NOT NULL,
	"used_articles" integer NOT NULL,
	"used_sources" integer NOT NULL,
	"tldr" text,
	"clustering_params" jsonb,
	"model_author" text,
	"created_at" timestamp DEFAULT CURRENT_TIMESTAMP NOT NULL
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "sources" (
	"id" serial PRIMARY KEY NOT NULL,
	"url" text NOT NULL,
	"name" text NOT NULL,
	"scrape_frequency" integer DEFAULT 2 NOT NULL,
	"paywall" boolean DEFAULT false NOT NULL,
	"category" text NOT NULL,
	"last_checked" timestamp,
	"do_initialized_at" timestamp,
	CONSTRAINT "sources_url_unique" UNIQUE("url")
);
--> statement-breakpoint
ALTER TABLE "articles" ADD CONSTRAINT "articles_source_id_sources_id_fk" FOREIGN KEY ("source_id") REFERENCES "public"."sources"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
CREATE INDEX "embeddingIndex" ON "articles" USING hnsw ("embedding" vector_cosine_ops);