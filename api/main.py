import logging
from contextlib import asynccontextmanager
from api.models.registry import load_detectors
from api.routes.prediction import router
from fastapi import FastAPI

logger = logging.getLogger("uvicorn.error")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading models")
    load_detectors()
    logger.info("Models loaded")

    yield  

app = FastAPI(lifespan=lifespan)

app.include_router(router)
