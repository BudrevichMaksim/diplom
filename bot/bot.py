import asyncio
import logging
import sys
from os import getenv
from pathlib import Path
import httpx
from string import Template

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message, ContentType


TOKEN: str = str(getenv("BOT_TOKEN"))
API_PATH: str = str(getenv("API_PATH"))

MAX_VOICE_DURATION: int = 10
MIN_VOICE_DURATION: int = 2

START_MESSAGE: str = f"Я создан помочь понять, общается ли с тобой реальный человек, перешли голосовое от {MIN_VOICE_DURATION} до {MAX_VOICE_DURATION} секунд и я отвечу в ближайшее время!"
REPLY_TO_VOICE: str = "Сообщение отправлено на анализ, ожидайте..."
DEFAULT_ANSWER: str = "Это сообщение не подходит.\nДля проверки мне нужно голосовое."

REPLY_PREDICTION = Template(
    "Результат анализа: $pred, с увереностью: $conf"
)


ERROR_TOO_LONG: str = f"Сообщение должно быть короче {MAX_VOICE_DURATION} секунд."
ERROR_TOO_SHORT: str = f"Сообщение должно быть длиннее {MIN_VOICE_DURATION} секунд."


dp = Dispatcher()


@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    """
    This handler receives messages with `/start` command
    """
    await message.answer(START_MESSAGE)

@dp.message(F.voice, F.voice.duration > MAX_VOICE_DURATION)
async def error_voice_too_long_handler(message: Message):
    await message.answer(ERROR_TOO_LONG)

@dp.message(F.voice, F.voice.duration < MIN_VOICE_DURATION)
async def error_voice_too_short_handler(message: Message):
    await message.answer(ERROR_TOO_SHORT)

@dp.message(F.voice)
async def voice_message_handler(message: Message, bot: Bot):
    """
    This handler recieves voice message
    """
    
    file_path = await download_audio(message, bot)

    await message.reply(REPLY_TO_VOICE)

    prediction = await get_prediction(str(file_path))

    await message.answer(REPLY_PREDICTION.substitute(
        pred=prediction["prediction"], 
        conf=prediction["confidence"]
        ))

@dp.message()
async def default_handler(message: Message):
    await message.answer(DEFAULT_ANSWER)


async def download_audio(message: Message, bot: Bot) -> Path:
    """
        This function downloads voice message
    """
    assert message.voice is not None
    destination_dir = Path("downloads/voices")
    destination_dir.mkdir(parents=True, exist_ok=True) 
    
    file_path = destination_dir / f"{message.voice.file_id}.ogg"
    
    await bot.download(message.voice, destination=file_path)

    return file_path

async def get_prediction(file_path: str):
    async with httpx.AsyncClient() as client:
        with open(file_path, "rb") as audio_file:
            response = await client.post(
                f"{API_PATH}/predict",
                files={"file": audio_file}
            )

        response.raise_for_status()
    
    return response.json()

async def main() -> None:
    # Initialize Bot instance with default bot properties which will be passed to all API calls
    bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))

    # And the run events dispatching
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())