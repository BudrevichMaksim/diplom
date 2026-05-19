from pathlib import Path

import httpx

from shared.schemas.response import PredictionResponse


class PredictionClient:
    """
    HTTP client responsible for communicating with the ML inference API.
    
    Handles asynchronous file uploads and maps raw JSON HTTP responses
    into structured Pydantic data schemas.
    """

    def __init__(self, api_path: str):
        """
        Initializes the client with the base URL of the inference API.

        Args:
            api_path (str): The base HTTP URL of the target API (e.g., 'http://localhost:8000').
        """
        self.api_path = api_path

    async def predict(
        self,
        file_path: Path
    ) -> PredictionResponse:
        """
        Sends an audio file to the API's prediction endpoint via multipart form-data.

        Args:
            file_path (Path): Path to the local audio file to be uploaded.

        Returns:
            PredictionResponse: The validated response schema containing the model's verdict.

        Raises:
            httpx.HTTPStatusError: If the backend API responds with an error code (4xx or 5xx).
            ValidationError: If the backend JSON response does not match the PredictionResponse schema.
        """
        # Set a 30-second timeout to accommodate larger files and model processing overhead
        async with httpx.AsyncClient(timeout=30) as client:
            with open(file_path, "rb") as audio_file:
                # Perform an asynchronous multipart/form-data POST request
                response = await client.post(
                    f"{self.api_path}/predict",
                    files={
                        "file": audio_file
                    }
                )

            # Raise an exception for 4xx or 5xx status codes automatically
            response.raise_for_status()

        # Parse JSON and validate it against the expected Pydantic model structure
        return PredictionResponse.model_validate(
            response.json()
        )
