import os
import glob
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm


def compute_log_class_weights(
    mask_paths: list[str], num_classes: int = 21, c: float = 1.02
) -> torch.Tensor:
    """Calcola i pesi delle classi con bilanciamento logaritmico leggendo direttamente

    le maschere grezze su disco (evitando i bias da data augmentation/cropping).

    Formula: w_c = 1 / log(c + f_c)

    Args:
        mask_paths: Lista dei path ai file delle maschere di ground truth (PNG/BMP/TIFF).
        num_classes: Numero totale di classi (es. 21 = 20 classi + background).
        c: Parametro di smoothing logaritmico (default: 1.02).

    Returns:
        torch.Tensor 1D di dimensione [num_classes] pronto per CrossEntropyLoss.
    """
    total_pixel_counts = np.zeros(num_classes, dtype=np.int64)

    print(f"Scansione di {len(mask_paths)} maschere originali...")
    for path in tqdm(mask_paths):
        # Carica la maschera originale senza trasformazioni
        mask = np.array(Image.open(path))

        # Conta i pixel per ogni ID di classe presente nell'immagine
        counts = np.bincount(mask.ravel(), minlength=num_classes)

        # Accumula i conteggi (considerando solo le classi da 0 a num_classes - 1)
        total_pixel_counts += counts[:num_classes]

    # 1. Frequenza relativa di ciascuna classe
    total_pixels = total_pixel_counts.sum()
    frequencies = total_pixel_counts / total_pixels

    # 2. Formula log-smoothed: w_c = 1 / log(c + f_c)
    weights = 1.0 / np.log(c + frequencies)

    # 3. Normalizzazione: mantiene la media dei pesi attorno a 1.0
    # per evitare di alterare la scala complessiva dei gradienti dell'optimizer
    weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32)


# 1. Recupera la lista dei path di tutte le maschere di train
# ERRORE CONCETTUALE QUI AL MOMENTO: PRESENTI TUTTE LE IMMAGINI PER CALCOLARE % e PESI,
# ANDREBBERO ESCLUSE LE IMMAGINI DI TEST
train_mask_paths = sorted(glob.glob("dataset/VOC_exercise/targets/*.png"))

# 2. Calcola i pesi esatti
class_weights = compute_log_class_weights(
    mask_paths=train_mask_paths, num_classes=21, c=1.01
)

# Stampa i pesi calcolati per verifica
print("Pesi per classe calcolati:")
print(f"Background (Classe 0): {class_weights[0].item():.4f}")
print(f"Altre classi (Media) : {class_weights[1:].mean().item():.4f}")
print("Vettore pesi completo: ")
for i in range(len(class_weights)):
    print(f"{class_weights[i].item():.2f}", end="  ")