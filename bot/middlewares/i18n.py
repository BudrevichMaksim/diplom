from typing import Callable, Dict, Any

from aiogram import BaseMiddleware
from aiogram.types import TelegramObject

from bot.locales.translator import translator_hub


class I18nMiddleware(BaseMiddleware):
    """
    Middleware responsible for extracting language codes and injecting 
    localized translation engines into the request lifecycle.
    
    Acts as an intermediary layer that checks incoming user profiles for 
    Telegram language codes, retrieves corresponding TranslatorRunners, 
    and appends them to the handler dependency context.
    """

    async def __call__(
        self, 
        handler: Callable[[TelegramObject, Dict[str, Any]], Any], 
        event: TelegramObject, 
        data: Dict[str, Any]
    ) -> Any:
        """
        Interceptors execution logic for all incoming Telegram events.

        Args:
            handler (Callable): The next middleware execution block or target router handler.
            event (TelegramObject): The raw incoming Telegram structure (Message, CallbackQuery, etc.).
            data (Dict[str, Any]): The dynamic contextual dictionary passed down the route chain.

        Returns:
            Any: The response object emitted by subsequent pipeline handlers.
        """
        # Default fallback locale if language cannot be determined or parsed
        locale = "en"

        # 'event_from_user' is automatically populated by Aiogram's context resolution
        user = data.get("event_from_user")

        if user and user.language_code:
            locale = user.language_code

        # Fetch the localized TranslatorRunner from our centralized Fluent hub
        translator = translator_hub.get_translator_by_locale(locale)

        # Inject the localization runner directly into the execution context.
        # This makes the runner accessible via parameter injection ('i18n: TranslatorRunner') inside handlers.
        data["i18n"] = translator

        # Forward execution to the next layer in the middleware/handler stack
        return await handler(event, data)
