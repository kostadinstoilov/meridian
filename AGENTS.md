# Meridian Development Guidelines

## Build Commands

- `pnpm dev` - Start all services in development mode
- `pnpm build` - Build all packages and apps
- `pnpm lint` - Run linting across all packages
- `pnpm typecheck` - Run TypeScript checks across all packages
- `pnpm format` - Format code with Prettier

### Backend (Cloudflare Workers)

- `cd apps/backend && pnpm test` - Run all tests
- `cd apps/backend && pnpm test [filename]` - Run single test file
- `cd apps/backend && pnpm lint` - Biome linting
- `cd apps/backend && pnpm lint:fix` - Auto-fix linting issues
- `cd apps/backend && pnpm typecheck` - TypeScript checks

### Frontend (Nuxt/Vue)

- `cd apps/frontend && pnpm lint` - ESLint checks
- `cd apps/frontend && pnpm lint:fix` - Auto-fix ESLint issues
- `cd apps/frontend && pnpm typecheck` - Nuxt TypeScript checks

### Database

- `cd packages/database && pnpm migrate` - Run database migrations
- `cd packages/database && pnpm generate` - Generate migration files
- `cd packages/database && pnpm studio` - Open Drizzle Studio

### ML Service (Python/FastAPI)

- `cd services/meridian-ml-service && uv venv` - Create virtual environment
- `cd services/meridian-ml-service && source .venv/bin/activate` - Activate virtual environment
- `cd services/meridian-ml-service && uv pip install -e .[dev]` - Install dependencies
- `cd services/meridian-ml-service && uvicorn meridian_ml_service.main:app --reload --host 0.0.0.0 --port 8080` - Run locally
- `cd services/meridian-ml-service && uv run ruff check . --fix` - Lint and fix
- `cd services/meridian-ml-service && uv run ruff format .` - Format code
- `cd services/meridian-ml-service && uv run mypy src/` - Type checking

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
- Use `uv` for package management and virtual environments
