from aiogram import Router
from aiogram.filters import CommandStart
from aiogram.types import Message

from fluentogram import TranslatorRunner


# Initialize a dedicated router module for welcoming users
router = Router()


@router.message(CommandStart())
async def start_handler(
    message: Message,
    i18n: TranslatorRunner
):
    """
    Handles the initial conversation entry point when a user sends /start.

    Fetches a localized welcoming message that explains the operational rules 
    of the bot, passing the minimum and maximum voice duration limits dynamically 
    to the localization template engine.

    Args:
        message (Message): The incoming Telegram message containing the /start command.
        i18n (TranslatorRunner): The locale-aware translation runner.
    """
    # Send the localized introduction message.
    # The parameters 'min' and 'max' are passed dynamically into the .ftl string variables.
    await message.answer(
        i18n.get(
            "start-message",
            min=2,
            max=10
        )
    )
