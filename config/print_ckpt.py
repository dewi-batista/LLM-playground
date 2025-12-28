import sys
import pprint
import torch

ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)

DROP = {
    "E_state_dict", "U_state_dict", "optimizer_state_dict",
    "rng_state_py", "rng_state_np", "rng_state_torch", "rng_state_cuda",
    "index_to_token",
}

print("index_to_token len:", len(ckpt.get("index_to_token", [])))
pprint.pp({k: v for k, v in ckpt.items() if k not in DROP})
