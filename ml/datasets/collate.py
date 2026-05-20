from typing import List, Dict
import torch
import torch.nn.functional as F


def spectrogram_collate_fn(
    batch: List[Dict[str, torch.Tensor]], max_frames: int = 256
) -> Dict[str, torch.Tensor]:
    """
    Collates a batch of acoustic spectrogram features by aligning their time axes.

    Truncates the temporal dimension (the final axis) of any tensor exceeding
    `max_frames`. Following truncation, it dynamically pads all samples in the
    batch on the right side to match the length of the longest remaining sequence.

    Args:
        batch (List[Dict[str, torch.Tensor]]): A list of data dictionaries where
            each item contains a "features" tensor and a "label" tensor.
        max_frames (int): The absolute ceiling limit allowed for the time axis.

    Returns:
        Dict[str, torch.Tensor]: A batched data dictionary containing:
            - "features": A stacked tensor of shape [Batch, Channels/Freq, Max_Time_In_Batch].
            - "label": A stacked tensor of shape [Batch].
    """
    features = [item["features"] for item in batch]
    labels = [item["label"] for item in batch]

    processed_features = []
    for feat in features:
        # Hard truncate along the time axis if it exceeds the max allowed window
        if feat.shape[-1] > max_frames:
            feat = feat[..., :max_frames]
        processed_features.append(feat)

    # Determine the dynamic padding ceiling based on the longest sample in this specific batch
    max_time = max(feat.shape[-1] for feat in processed_features)

    # Pad only the right side of the final (temporal) dimension
    padded = [
        F.pad(feat, (0, max_time - feat.shape[-1])) for feat in processed_features
    ]

    return {
        "features": torch.stack(padded),
        "label": torch.stack(labels),
    }
