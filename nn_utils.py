import numpy as np

import torch
from torch.utils.data import WeightedRandomSampler
import torch.nn.functional as F


def compute_class_weights(labels, num_classes, ignore_index, method="median_freq"):
    labels_np = np.asarray(labels).flatten()
    labels_np = labels_np[labels_np != ignore_index]

    # ── count samples per class ───────────────────────────────────────────────
    counts = np.array([(labels_np == c).sum() for c in range(num_classes)], dtype=np.float64)
    total  = counts.sum()
    freq   = counts / total

    # print("Class counts:", {c: int(counts[c]) for c in range(num_classes)})

    if method == "inverse_freq":
        weights = 1.0 / (freq + 1e-6)
    elif method == "inverse_sqrt":
        weights = 1.0 / (np.sqrt(freq) + 1e-6)
    elif method == "median_freq":
        # w_c = median(freq) / freq_c (SegNet paper)
        weights = np.median(freq) / (freq + 1e-6)
    elif method == "effective_num":
        # from "Class-Balanced Loss Based on Effective Number of Samples" CVPR 2019
        beta    = (total - 1.0) / total
        weights = (1.0 - beta) / (1.0 - np.power(beta, counts + 1e-6))
    else:
        raise ValueError(f"Unknown method '{method}'.")

    weights = weights / weights.sum() * num_classes
    # print(f"Class weights ({method}):", {c: round(weights[c], 4) for c in range(num_classes)})

    return torch.tensor(weights, dtype=torch.float32)


def build_weighted_sampler(dataset):
    weights = []
    ignore = dataset.ignore_index

    for idx in range(len(dataset.distr_data)):
        cur_x = dataset.distr_data[idx][0]
        cur_y = dataset.distr_data[idx][1]
        patch_mask = dataset.train_mask[cur_x:cur_x + dataset.patch_size, cur_y:cur_y + dataset.patch_size]
        valid_pixels = patch_mask[patch_mask != ignore]

        assert len(valid_pixels) != 0, "No valid pixels in mask"

        # weight = inverse of the most frequent class in this patch
        counts = np.bincount(valid_pixels, minlength=dataset.num_classes)
        freq = counts / (counts.sum() + 1e-9)

        # find the rarest class present in this patch
        present = freq[freq > 0]
        w = 1.0 / (present.min() + 1e-6)
        weights.append(w)

    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )


def compute_patch_difficulties(model, dataset, batch_size=128):
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    all_losses = []
    with torch.no_grad():
        for imgs, masks, _ in loader:
            imgs  = imgs.cuda()
            masks = masks.cuda()

            logits, _, _, _ = model(imgs.permute(1, 0, 2, 3, 4))
            loss = F.cross_entropy(logits, masks, ignore_index=dataset.ignore_index, reduction='none')  # (B, H, W)

            valid = (masks != dataset.ignore_index).float()
            patch_loss = (loss * valid).sum(dim=[1, 2]) / (valid.sum(dim=[1, 2]) + 1e-9)
            all_losses.append(patch_loss.cpu())

    return torch.cat(all_losses).numpy()


def update_sampler(model, dataset, temperature=1.0):
    """
    Build a new sampler weighted by current model difficulty.
    temperature > 1 = more uniform, temperature < 1 = more focused on hard patches
    """
    difficulties = compute_patch_difficulties(model, dataset)

    # softmax over difficulties to get valid probability distribution
    d = difficulties / (difficulties.max() + 1e-9)
    weights = np.exp(d / temperature)
    weights = weights / weights.sum()

    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

