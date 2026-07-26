# ── Stage 1: Build ──────────────────────────────────────────────────
FROM node:26-alpine AS builder

RUN apk add --no-cache python3 make g++ && npm install -g pnpm@10

WORKDIR /app

COPY pnpm-workspace.yaml pnpm-lock.yaml tsconfig.base.json ./
COPY packages/core/package.json packages/core/tsconfig.json packages/core/
COPY packages/pipeline/package.json packages/pipeline/tsconfig.json packages/pipeline/
COPY packages/storage-embed/package.json packages/storage-embed/tsconfig.json packages/storage-embed/
COPY packages/storage-remote/package.json packages/storage-remote/tsconfig.json packages/storage-remote/
COPY packages/visual/package.json packages/visual/tsconfig.json packages/visual/

RUN pnpm install --frozen-lockfile

COPY packages/core/src packages/core/src/
COPY packages/pipeline/src packages/pipeline/src/
COPY packages/storage-embed/src packages/storage-embed/src/
COPY packages/storage-remote/src packages/storage-remote/src/
COPY packages/visual/src packages/visual/src/

RUN pnpm run --filter @agentix-e/causality-analyzer-core build
RUN pnpm run --filter @agentix-e/causality-analyzer-pipeline build

# ── Stage 2: Production ──────────────────────────────────────────────
FROM node:26-alpine

RUN apk add --no-cache python3 make g++ && npm install -g pnpm@10
RUN addgroup -S ca && adduser -S ca -G ca

WORKDIR /app
RUN chown ca:ca /app

COPY --from=builder --chown=ca:ca /app/packages/core/package.json packages/core/
COPY --from=builder --chown=ca:ca /app/packages/core/dist packages/core/dist/
COPY --from=builder --chown=ca:ca /app/packages/pipeline/package.json packages/pipeline/
COPY --from=builder --chown=ca:ca /app/packages/pipeline/dist packages/pipeline/dist/
COPY --from=builder --chown=ca:ca /app/pnpm-workspace.yaml ./
COPY --from=builder --chown=ca:ca /app/pnpm-lock.yaml ./

# Install production dependencies only
RUN pnpm install --prod --frozen-lockfile

# Copy entrypoint script
COPY docker-entrypoint.js ./

ENV NODE_ENV=production
ENV PORT=3000

USER ca

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD node -e "require('http').get('http://localhost:3000/health',r=>{process.exit(r.statusCode===200?0:1)})"

ENTRYPOINT ["node", "docker-entrypoint.js"]
