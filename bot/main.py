import asyncio
import logging
import sys

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from bot.core.dependencies import Dependencies

from bot.handlers.start import router as start_router
from bot.handlers.voice import create_voice_router
from bot.handlers.default import router as default_router

from bot.middlewares.i18n import I18nMiddleware


async def main():
    """
    Main asynchronous startup routine for the Telegram Bot.
    
    Wires up the DI container, configures core framework primitives (Bot, Dispatcher),
    registers global interceptors (Middlewares), attaches isolated execution routes,
    and initiates the polling runtime engine.
    """
    # 1. Initialize the Dependency Injection container (Loads config & services)
    deps = Dependencies()

    # 2. Instantiate the low-level Telegram Bot API client connection wrapper
    bot = Bot(
        token=deps.settings.bot_token,
        default=DefaultBotProperties(
            parse_mode=ParseMode.HTML  # Enables global support for safe HTML markdown elements
        )
    )

    # 3. Instantiate the root Dispatcher responsible for processing incoming updates
    dp = Dispatcher()

    # 4. Attach global internationalization (i18n) middleware to all message events
    dp.message.middleware(I18nMiddleware())

    # 5. Connect application sub-routers to the root execution stack
    # NOTE: Order matters heavily here!
    dp.include_router(start_router)  # 1st Priority: Explicit commands (/start)

    dp.include_router(               # 2nd Priority: Voice processing & ML inference
        create_voice_router(
            prediction_service=deps.prediction_service,
            file_service=deps.file_service,
            lock_service=deps.lock_service,
            settings=deps.settings
        )
    )

    dp.include_router(default_router) # 3rd Priority: Catch-all fallback guardrail

    # 6. Wipe any pending updates waiting on Telegram clouds and begin long-polling
    await dp.start_polling(bot)


if __name__ == "__main__":
    # Configure production-ready system logging layout directed to standard output streams
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Launch the asynchronous runtime loop
    asyncio.run(main())
