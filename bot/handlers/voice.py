import logging

from aiogram import Router
from aiogram import F
from aiogram import Bot
from aiogram.types import Message
from fluentogram import TranslatorRunner

# Initialize a standard logger for tracking execution anomalies
logger = logging.getLogger(__name__)


def create_voice_router(prediction_service, file_service, lock_service, settings):
    """
    Factory function that builds and configures the voice processing router.

    By utilizing a closure/factory pattern, this function cleanly injects required
    business logic services and global settings straight into the handler scopes
    without relying on global state variables.

    Args:
        prediction_service (PredictionService): Service to communicate with the ML API.
        file_service (FileService): Service dealing with Telegram file downloading.
        lock_service (LockService): Concurrency control tracking active user operations.
        settings (Settings): Global configuration holding constraints like duration boundaries.

    Returns:
        Router: A fully configured Aiogram Router ready for dispatcher registration.
    """
    router = Router()

    @router.message(F.voice, F.voice.duration > settings.max_voice_duration)
    async def too_long_handler(message: Message, i18n: TranslatorRunner):
        """Filters out voice messages exceeding the maximum allowed length."""
        await message.answer(
            i18n.get("error-too-long", max=settings.max_voice_duration)
        )

    @router.message(F.voice, F.voice.duration < settings.min_voice_duration)
    async def too_short_handler(message: Message, i18n: TranslatorRunner):
        """Filters out voice messages falling short of the minimum required length."""
        await message.answer(
            i18n.get("error-too-short", min=settings.min_voice_duration)
        )

    @router.message(F.voice)
    async def voice_handler(message: Message, bot: Bot, i18n: TranslatorRunner):
        """
        Processes valid voice notes, downloads them, and orchestrates ML analysis.

        Employs user-level non-blocking locks to ensure a single user cannot trigger
        multiple concurrent heavy inference workloads simultaneously.
        """
        # Safety assertion to ensure the message contains user context
        assert message.from_user is not None

        user_id = message.from_user.id
        lock = lock_service.get_lock(user_id)

        # Early exit if the user has an unresolved analysis running
        if lock.locked():
            await message.answer(i18n.get("error-sending-blocked"))
            return

        # Acquire lock to isolate the user's execution stream
        async with lock:
            try:
                # 1. Download the voice message from Telegram's servers
                file_path = await file_service.download_voice(message.voice, bot)

                # 2. Notify user that processing has initiated
                await message.answer(i18n.get("reply-to-voice"))

                # 3. Request inference via the client network boundary
                prediction = await prediction_service.analyze(file_path)

                # 4. Reply with localized classification and rounded confidence intervals
                await message.reply(
                    i18n.get(
                        "prediction-result",
                        pred=prediction.prediction,
                        conf=round(prediction.confidence, 2),
                    )
                )

            except Exception as e:
                # Log any server errors, file issues, or HTTP timeouts gracefully
                logging.exception(
                    f"Error handling voice message for user {user_id}: {e}"
                )
                await message.answer(i18n.get("error-server-not-responding"))

    return router
