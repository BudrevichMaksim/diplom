from fluentogram import TranslatorHub
from fluentogram.storage.file import FileStorage

# Define the file system storage mapping to your localization assets directory.
# This folder is expected to contain locale subdirectories (e.g., /ru, /en) with .ftl files.
storage = FileStorage("bot/locales/lang")

# Initialize the central Translation Hub.
# The dictionary defines the fallback hierarchy for available language keys.
translator_hub = TranslatorHub(
    {
        "ru": ("ru",),  # If 'ru' is requested, resolve using 'ru' files
        "en": ("en",),  # If 'en' is requested, resolve using 'en' files
    },
    storage=storage,
    root_locale="en",  # Fallback locale used if a requested key is missing in another language
)
