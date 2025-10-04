from typing import Annotated, Union
import os

import asyncio
from functools import lru_cache

from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.status import HTTP_403_FORBIDDEN

from .config import settings
from .embeddings import ModelComponents, load_embedding_model

# Global lock for model loading
_model_lock = asyncio.Lock()
_model_instance: Union[ModelComponents, None] = None


async def get_embedding_model() -> ModelComponents:
    """FastAPI dependency to get the loaded embedding model components in a thread-safe way."""
    global _model_instance

    if _model_instance is not None:
        return _model_instance

    async with _model_lock:
        # double-check pattern to avoid race conditions
        if _model_instance is not None:
            return _model_instance

        try:
            _model_instance = load_embedding_model()
            return _model_instance
        except Exception as e:
            # Consider how to handle model loading failure more gracefully in API
            # Maybe return HTTP 503 Service Unavailable?
            # Use logging instead of print to avoid leaking secrets
            import logging

            logging.getLogger(__name__).exception("FATAL: Could not provide embedding model")
            raise  # Let FastAPI handle internal server error for now


ModelDep = Annotated[ModelComponents, Depends(get_embedding_model)]

# Use HTTPBearer to parse Authorization header cleanly, but keep auto_error=False so
# older tests that expect auth disabled when no token is set still work.
bearer_scheme = HTTPBearer(auto_error=False)


async def verify_token(creds: HTTPAuthorizationCredentials | None = Security(bearer_scheme)) -> None:
    # Read API token dynamically from environment to respect test monkeypatching
    env_token = os.getenv("API_TOKEN")
    if env_token is None:
        return  # auth is disabled if no token is configured

    # creds may be None (no header) or have .credentials
    if creds is None or creds.credentials != env_token:
        raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Invalid or missing API token")
