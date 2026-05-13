from api.feature_extractors.base import BaseFeatureExtractor
from api.feature_extractors.mel import MelSpectrogramExtractor


def get_extractor(name: str) -> BaseFeatureExtractor:
    """
    Factory function to retrieve a feature extractor instance by name.

    Args:
        name (str): The identifier for the extractor (e.g., "mel").

    Returns:
        BaseFeatureExtractor: An instance of the requested extractor.

    Raises:
        KeyError: If the provided name is not registered in the extractors dictionary.
    """
    extractors = {"mel": MelSpectrogramExtractor()}

    return extractors[name]
