import logging
from contextlib import asynccontextmanager
from api.models.registry import load_detector
from api.routes.prediction import router
from fastapi import FastAPI

# Use the Uvicorn error logger to ensure logs appear in the console during server runtime
logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Asynchronous context manager for managing the application lifecycle.
    
    This handles the startup logic (loading heavy ML models) and ensures
    resources are properly initialized before the server starts accepting traffic.
    """
    logger.info("Application startup: Initializing spoofing detection models...")

    # Load the specific detector defined in settings and store it in app.state.
    # This makes the model instance accessible to all request handlers via dependencies.
    app.state.detector = load_detector()

    logger.info("Application startup: Models loaded and ready for inference.")
    
    yield  # The application serves requests in this state

    # Logic after 'yield' runs during application shutdown
    logger.info("Application shutdown: Cleaning up resources.")


# Initialize the FastAPI app with the lifespan context manager
app = FastAPI(lifespan=lifespan)

# Register the prediction endpoints
app.include_router(router)