from pathlib import Path

from bot.clients.prediction_client import PredictionClient
from shared.schemas.response import PredictionResponse


class PredictionService:
    """
    Service layer responsible for handling audio analysis workflows within the bot.

    Acts as an abstraction layer between the bot handlers and the underlying 
    API client, ensuring the business logic remains decoupled from network protocols.
    """

    def __init__(
        self,
        prediction_client: PredictionClient
    ):
        """
        Initializes the PredictionService with a dedicated HTTP client.

        Args:
            prediction_client (PredictionClient): The client used to communicate 
                                                    with the ML inference API.
        """
        self.prediction_client = prediction_client

    async def analyze(
        self,
        file_path: Path
    ) -> PredictionResponse:
        """
        Submits a local audio file for spoofing analysis.

        Args:
            file_path (Path): Path to the audio file saved locally on the bot's host.

        Returns:
            PredictionResponse: A validated shared response schema containing the 
                                prediction verdict and confidence score.
        """
        # Forward the local file path directly to the API client for upload and inference
        return await self.prediction_client.predict(
            file_path
        )
