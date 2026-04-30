import asyncio
import logging
import sys
from os import getenv
from pathlib import Path

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message, ContentType


TOKEN: str = str(getenv("BOT_TOKEN"))
START_MESSAGE: str = "Я создан помочь понять, общается ли с тобой реальный человек, перешли голосовое и я отвечу в ближайшее время!"
REPLY_TO_VOICE: str = "Сообщение отправлено на анализ, ожидайте..."

dp = Dispatcher()


@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    """
    This handler receives messages with `/start` command
    """
    await message.answer(START_MESSAGE)


@dp.message(F.voice)
async def voice_message_handler(message: Message, bot: Bot):
    """
    This handler recieves voice message
    """
    assert message.voice is not None

    destination_dir = Path("downloads/voices")
    destination_dir.mkdir(parents=True, exist_ok=True) 
    
    file_path = destination_dir / f"{message.voice.file_id}.ogg"
    
    await bot.download(message.voice, destination=file_path)

    await message.reply(REPLY_TO_VOICE)


async def main() -> None:
    # Initialize Bot instance with default bot properties which will be passed to all API calls
    bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))

    # And the run events dispatching
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())