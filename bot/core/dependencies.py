from bot.config.settings import Settings

from bot.clients.prediction_client import PredictionClient

from bot.services.prediction_service import PredictionService
from bot.services.file_service import FileService
from bot.services.lock_service import LockService


class Dependencies:
    """
    Dependency Injection Container for the Telegram bot application.

    This class handles the initialization, lifecycle, and wiring of all
    core services, clients, and configurations required by the bot handlers.
    """

    def __init__(self):
        """
        Initializes and wires all application dependencies.

        Loads settings first, builds the network client infrastructure, and
        injects necessary components into downstream services to prevent tight coupling.
        """
        # 1. Load application configuration (automatically reads .env)
        self.settings = Settings()

        # 2. Initialize the low-level HTTP network infrastructure
        prediction_client = PredictionClient(api_path=self.settings.api_path)

        # 3. Inject the client into the business logic service layer
        self.prediction_service = PredictionService(prediction_client)

        # 4. Instantiate local workspace I/O utility managers
        self.file_service = FileService()

        # 5. Instantiate the concurrency control mechanism for user request throttling
        self.lock_service = LockService()
