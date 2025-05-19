ALTER TABLE "articles" ALTER COLUMN "status" SET DATA TYPE text;--> statement-breakpoint
ALTER TABLE "articles" ALTER COLUMN "status" SET DEFAULT 'PENDING_FETCH'::text;--> statement-breakpoint
DROP TYPE "public"."article_status";--> statement-breakpoint
CREATE TYPE "public"."article_status" AS ENUM('PENDING_FETCH', 'CONTENT_FETCHED', 'PROCESSED', 'SKIPPED_PDF', 'FETCH_FAILED', 'RENDER_FAILED', 'EMBEDDING_FAILED', 'R2_UPLOAD_FAILED', 'SKIPPED_TOO_OLD');--> statement-breakpoint
ALTER TABLE "articles" ALTER COLUMN "status" SET DEFAULT 'PENDING_FETCH'::"public"."article_status";--> statement-breakpoint
ALTER TABLE "articles" ALTER COLUMN "status" SET DATA TYPE "public"."article_status" USING "status"::"public"."article_status";--> statement-breakpoint
ALTER TABLE "articles" ADD COLUMN "word_count" integer;--> statement-breakpoint
ALTER TABLE "articles" DROP COLUMN "language";--> statement-breakpoint
ALTER TABLE "articles" DROP COLUMN "primary_location";--> statement-breakpoint
ALTER TABLE "articles" DROP COLUMN "completeness";--> statement-breakpoint
ALTER TABLE "articles" DROP COLUMN "content_quality";--> statement-breakpoint
ALTER TABLE "articles" DROP COLUMN "event_summary_points";--> statement-breakpoint
ALTER TABLE "articles" DROP COLUMN "thematic_keywords";--> statement-breakpoint
ALTER TABLE "articles" DROP COLUMN "topic_tags";--> statement-breakpoint
ALTER TABLE "articles" DROP COLUMN "key_entities";--> statement-breakpoint
ALTER TABLE "articles" DROP COLUMN "content_focus";--> statement-breakpoint
DROP TYPE "public"."article_completeness";--> statement-breakpoint
DROP TYPE "public"."article_content_quality";