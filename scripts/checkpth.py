import torch

path = r"C:\Users\14259\Downloads\drive-download-20260410T043949Z-3-001\best_model_id_04.pth（副本）.zip"
ckpt = torch.load(path, map_location="cpu")
print(type(ckpt))
if isinstance(ckpt, dict):
    print(ckpt.keys())