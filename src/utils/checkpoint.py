import torch


def load_checkpoint(path: str, model: torch.nn.Module, device: torch.device) -> dict:
    """Load a checkpoint and restore model weights. Returns the full checkpoint dict.

    Handles both the current format (dict with 'model' key) and the legacy format
    (bare state_dict saved before resume support was added).
    """
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
        ckpt = {'model': ckpt}
    return ckpt
