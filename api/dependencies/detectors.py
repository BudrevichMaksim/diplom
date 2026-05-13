from fastapi import Request
from api.models.base import BaseDetector

async def get_detector(request: Request) -> BaseDetector:
    """
    Dependency function to retrieve the global detector instance from the application state.

    This ensures that the model is loaded only once during application startup 
    and shared across all incoming requests, optimizing memory usage and performance.

    Args:
        request (Request): The incoming FastAPI request object.

    Returns:
        BaseDetector: The initialized detector instance stored in the app state.
        
    Note:
        The detector must be assigned to 'app.state.detector' during the 
        application startup event for this dependency to work correctly.
    """
    return request.app.state.detector