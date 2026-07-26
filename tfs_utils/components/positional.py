import torch


def sinusoidal_positional_encoding(seq_len: int, d_model: int, device: torch.device):
    pos = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)  # (T, 1)
    i = torch.arange(d_model, device=device, dtype=torch.float32)  # (d,)
    div = torch.pow(10_000.0, (2 * (i // 2)) / d_model)
    angles = pos / div

    pe = torch.zeros_like(angles)
    pe[:, 0::2] = torch.sin(angles[:, 0::2])
    pe[:, 1::2] = torch.cos(angles[:, 1::2])
    return pe


POS_ENCODINGS = {
    "sinusoidal": sinusoidal_positional_encoding,
}
