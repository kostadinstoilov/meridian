CREATE TYPE "public"."ingested_item_status" AS ENUM('NEW', 'PENDING_PROCESSING', 'PROCESSED', 'FAILED_FETCH', 'FAILED_PROCESSING', 'SKIPPED_PDF', 'SKIPPED_TOO_OLD');--> statement-breakpoint
CREATE TYPE "public"."source_type" AS ENUM('RSS');--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "data_sources" (
	"id" serial PRIMARY KEY NOT NULL,
	"name" text NOT NULL,
	"source_type" "source_type" NOT NULL,
	"config" jsonb NOT NULL,
	"config_version_hash" text,
	"publisher_id" integer,
	"scrape_frequency_minutes" integer DEFAULT 240 NOT NULL,
	"last_checked" timestamp,
	"do_initialized_at" timestamp,
	"created_at" timestamp DEFAULT now() NOT NULL,
	"updated_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "ingested_items" (
	"id" bigserial PRIMARY KEY NOT NULL,
	"item_id_from_source" text NOT NULL,
	"raw_data_r2_key" text NOT NULL,
	"display_title" text,
	"url_to_original" text NOT NULL,
	"published_at" timestamp,
	"status" "ingested_item_status" DEFAULT 'NEW',
	"content_body_r2_key" text,
	"content_body_text" text,
	"word_count" integer,
	"analysis_payload" jsonb,
	"source_specific_metadata" jsonb,
	"used_browser" boolean,
	"embedding" vector(384),
	"fail_reason" text,
	"data_source_id" integer NOT NULL,
	"processed_at" timestamp,
	"ingested_at" timestamp DEFAULT CURRENT_TIMESTAMP,
	CONSTRAINT "ingested_items_url_to_original_unique" UNIQUE("url_to_original"),
	CONSTRAINT "uniqueSourceItem" UNIQUE("data_source_id","item_id_from_source")
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "newsletter" (
	"id" serial PRIMARY KEY NOT NULL,
	"email" text NOT NULL,
	"created_at" timestamp DEFAULT CURRENT_TIMESTAMP,
	CONSTRAINT "newsletter_email_unique" UNIQUE("email")
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "publishers" (
	"id" serial PRIMARY KEY NOT NULL,
	"name" text NOT NULL,
	"base_url" text,
	"created_at" timestamp DEFAULT now() NOT NULL
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
ALTER TABLE IF EXISTS "data_sources" ADD CONSTRAINT "data_sources_publisher_id_publishers_id_fk" FOREIGN KEY ("publisher_id") REFERENCES "public"."publishers"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE IF EXISTS "ingested_items" ADD CONSTRAINT "ingested_items_data_source_id_data_sources_id_fk" FOREIGN KEY ("data_source_id") REFERENCES "public"."data_sources"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "embeddingIndex" ON "ingested_items" USING hnsw ("embedding" vector_cosine_ops);