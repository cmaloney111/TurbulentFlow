from pathlib import Path
import torch
import numpy as np
import os

overwrite = None

for pth_file in Path(os.getcwd()).glob("*.pth"):
    npz_file = pth_file.with_suffix(".npz")
    if npz_file.exists():
        if overwrite is None:
            overwrite = input(f"Overwrite NumPy files? [y/n]")
        if overwrite.lower() != "y":
            continue

    print(f"Converting {pth_file} to {npz_file}")
    state_dict = torch.load(pth_file, map_location=torch.device('cpu'))["model_state_dict"]
    np.savez_compressed(npz_file, **{
        key: value.cpu().numpy()
        for key, value in state_dict.items()
    })