# Meridian Repository Overview

## Project Description
Meridian is an AI-powered personal intelligence agency that delivers presidential-level daily briefings. It scrapes hundreds of news sources, analyzes stories with AI (Gemini), and provides concise, personalized intelligence briefs with context and analysis beyond headlines.

## File Structure
- **apps/**: Main applications
  - `backend/`: Cloudflare Workers backend with Hono framework
  - `briefs/`: Brief generation application
  - `frontend/`: Nuxt 3 frontend with Vue 3 and Tailwind CSS
- **packages/**: Shared packages
  - `database/`: PostgreSQL database schema with Drizzle ORM
- **services/**: External services
  - `meridian-ml-service/`: Python ML service for embeddings and clustering
- **Root**: Turborepo monorepo configuration with pnpm workspace

## Tech Stack
- **Infrastructure**: Turborepo, Cloudflare Workers/Workflows/Pages
- **Backend**: Hono, TypeScript, PostgreSQL, Drizzle
- **AI/ML**: Gemini models, multilingual-e5-small embeddings, UMAP, HDBSCAN
- **Frontend**: Nuxt 3, Vue 3, Tailwind CSS
- **Package Manager**: pnpm v9.15+
- **Python**: ML service with uv for dependency management

## Development Commands
```bash
# Install dependencies
pnpm install          # Install all Node.js dependencies
uv sync               # Install Python dependencies for ML service

# Development
pnpm dev              # Run all apps in dev mode
pnpm build            # Build all apps
pnpm lint             # Lint all packages
pnpm format           # Format code with Prettier
pnpm typecheck        # Type check all packages

# Database
pnpm --filter @meridian/database migrate  # Run database migrations
```

## Running ML Service
```bash
# Navigate to ML service directory
cd services/meridian-ml-service

# Activate venv (if present)
source .venv/bin/activate

# Install dependencies
uv pip install -e .[dev]

# run with uvicorn for development
uvicorn meridian_ml_service.main:app --reload --host 0.0.0.0 --port 8080
```

## Environment Setup
1. **Node.js Environment**: Uses pnpm workspace with Turborepo
2. **Python Environment**: Uses uv for dependency management in ML service
3. **Database**: PostgreSQL with Drizzle ORM for schema management
4. **Cloudflare**: Requires Wrangler CLI for deployment

## Dependency Management
- **Node.js**: pnpm workspaces with hoisting enabled
- **Python**: uv for fast, isolated dependency management. Dependencies are defined in pyproject.toml
- **Database**: Drizzle migrations for schema versioning
- **Cloudflare**: Wrangler for deployment and local development

## Setup Requirements
- Node.js v22+
- pnpm v9.15+
- Python 3.10+
- PostgreSQL
- Cloudflare account
- Google AI API key

## Key Workflows
1. **Scraping**: Cloudflare Workers fetch RSS feeds and store metadata
2. **Processing**: Extract content and analyze with Gemini for relevance
3. **Brief Generation**: Python ML service clusters articles and generates analysis
4. **Frontend**: Nuxt app displays briefs via API

## Code Style Guidelines

### Formatting

- Use Prettier config: single quotes, semicolons, 2-space indent, 120 char width
- Backend uses Biome for linting (formatter disabled, organize imports enabled)
- Frontend uses ESLint with Nuxt config
- ML Service uses Ruff for linting and formatting (88 char width, double quotes)

### Import Style

- Use absolute imports with `@meridian/database` for workspace packages
- Group imports: external libraries first, then internal modules
- Use type-only imports where possible: `import type { ... }`

### TypeScript

- Strict TypeScript enabled across all packages
- Use `neverthrow` for Result-based error handling in backend
- Zod for runtime validation and type safety
- Explicit return types for public functions

### Error Handling

- Backend: Use `tryCatchAsync` wrapper for promise-based error handling
- Return Result<T, unknown> types from async operations
- Frontend: Use Vue's error boundaries and proper async handling

### Naming Conventions

- Files: kebab-case for utilities, PascalCase for components
- Functions: camelCase with descriptive names
- Constants: UPPER_SNAKE_CASE for configuration values
- Types: PascalCase for interfaces and type aliases

### Vue/Nuxt Specific

- Use Composition API with `<script setup>`
- Composables should be prefixed with `use`
- Auto-imports enabled for Vue and Nuxt utilities
- Use Tailwind CSS for styling with Radix UI colors

### Python/FastAPI Specific

- Use FastAPI with Pydantic v2 for data validation
- Follow PEP 8 style with Ruff formatting (88 char width)
- Use type hints consistently with Mypy for static checking

