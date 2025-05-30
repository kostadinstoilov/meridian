ALTER TYPE "public"."ingested_item_status" ADD VALUE 'FAILED_RENDER' BEFORE 'FAILED_FETCH';--> statement-breakpoint
ALTER TYPE "public"."ingested_item_status" ADD VALUE 'FAILED_EMBEDDING' BEFORE 'SKIPPED_PDF';--> statement-breakpoint
ALTER TYPE "public"."ingested_item_status" ADD VALUE 'FAILED_R2_UPLOAD' BEFORE 'SKIPPED_PDF';