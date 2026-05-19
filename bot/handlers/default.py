from aiogram import Router
from aiogram.types import Message

from fluentogram import TranslatorRunner


# Initialize a dedicated router module for managing catch-all fallback text messages
router = Router()


@router.message()
async def default_handler(
    message: Message,
    i18n: TranslatorRunner
):
    """
    Acts as a catch-all fallback message handler.

    This handler triggers for any user message that didn't match previous routers 
    or strict criteria (like specific commands or media filters). It fetches a 
    localized response key via Fluentogram and replies to the user.

    Args:
        message (Message): The incoming Telegram message object.
        i18n (TranslatorRunner): The localized translation runner injected 
                                    by the middleware layer.
    """
    # Send the localized generic response back to the user
    await message.answer(
        i18n.get(
            "default-answer"
        )
    )
