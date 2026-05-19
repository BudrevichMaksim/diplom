from pathlib import Path

from aiogram import Bot


class FileService:
    """
    Service responsible for managing local disk workspace and file downloads
    from Telegram servers.

    Encapsulates all path mechanics and ensures that media files are organized
    consistently on the file system.
    """

    def __init__(self, upload_dir: str = "downloads/voices"):
        """
        Initializes the FileService and creates the target storage directory.

        Args:
            upload_dir (str): Relative or absolute filesystem path where downloaded
                                audio files will be kept. Defaults to 'downloads/voices'.
        """
        self.upload_dir = Path(upload_dir)

        # Ensure the storage workspace folder exists immediately upon service initialization
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    async def download_voice(self, file_data, bot: Bot) -> Path:
        """
        Downloads a voice file from Telegram's cloud storage onto the local disk.

        Uses Telegram's guaranteed unique file identifier ('file_id') as the
        filename to safely bypass local race conditions or overwrites.

        Args:
            file_data (Voice): The Aiogram voice media metadata object containing the file parameters.
            bot (Bot): The operational, authenticated Telegram Bot client instance.

        Returns:
            Path: The explicit, resolved local Path object pointing to the downloaded .ogg asset.
        """
        # Formulate a bulletproof local destination using Telegram's unique file token
        file_path = self.upload_dir / f"{file_data.file_id}.ogg"

        # Asynchronously download the asset via HTTP streaming straight to the disk path
        await bot.download(file_data, destination=file_path)

        return file_path
