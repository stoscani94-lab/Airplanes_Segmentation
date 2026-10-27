import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassConfusionMatrix

from dataset import SegmentationDataModule
from architecture import get_custom_baseline, get_unet, get_deeplabv3plus, get_transformer_seg
from lightning_module import SegmentationTask

def main():
    # 1. PARAMETRI E SETUP
    # Inserisci il percorso esatto del miglior file .ckpt generato da Lightning
    # (es: "checkpoints/best-model-epoch=29-val_loss=1.45.ckpt")
    print("Inserisci il percorso del checkpoint (compreso di .ckpt): ")
    ckpt_path = input() # Esempio: "checkpoints/best-model-epoch=29-val_loss=1.45.ckpt"
    num_classes = 21
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Nomi delle classi (assumendo il formato Pascal VOC 20 + Background)
    # Modificali se il tuo dataset ha classi diverse
    class_names = ["Background", "Aeroplane", "Bicycle", "Bird", "Boat", "Bottle", 
                   "Bus", "Car", "Cat", "Chair", "Cow", "DiningTable", "Dog", 
                   "Horse", "Motorbike", "Person", "PottedPlant", "Sheep", 
                   "Sofa", "Train", "TvMonitor"]

    # 2. CARICAMENTO DATI
    datamodule = SegmentationDataModule(data_dir="dataset/VOC_exercise/", batch_size=4)
    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()

    # 3. CARICAMENTO MODELLO
    # net = get_custom_baseline(num_classes=num_classes)
    # net = get_unet(num_classes=21)
    # net = get_deeplabv3plus(num_classes=21)
    net = get_transformer_seg(num_classes=21)
    model = SegmentationTask.load_from_checkpoint(ckpt_path, model=net)
    model.to(device)
    model.eval()

    # 4. INIZIALIZZAZIONE METRICHE
    # average="none" ci restituisce un array con l'IoU specifica per ogni singola classe
    iou_metric = MulticlassJaccardIndex(num_classes=num_classes, average="none").to(device)
    conf_mat_metric = MulticlassConfusionMatrix(num_classes=num_classes, normalize="true").to(device)

    # 5. LOOP DI TEST SULL'INTERO TEST SET
    print("Calcolo delle metriche sull'intero test set in corso...")
    last_images, last_masks, last_preds = None, None, None
    with torch.no_grad():
        for batch in test_loader:
            images, masks = batch
            images, masks = images.to(device), masks.to(device)
            
            # Non applichiamo augmentation nel test, ma normalizziamo i tensori
            # Assicurati di usare le stesse trasformazioni del val_transforms nel tuo lightning_module
            images, masks = model.val_transforms(images, masks)
            
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            
            # Accumula i risultati per le metriche
            iou_metric.update(preds, masks)
            conf_mat_metric.update(preds, masks)
            
            # Conserviamo l'ultimo batch per visualizzarlo graficamente
            if last_images is not None:
                almost_last_images, almost_last_masks, almost_last_preds = last_images, last_masks, last_preds
            last_images, last_masks, last_preds = images, masks, preds


    # 6. ESTRAZIONE E STAMPA METRICHE
    class_ious = iou_metric.compute().cpu()
    conf_matrix = conf_mat_metric.compute().cpu()
    
    print("\n=== RISULTATI TEST SET ===")
    print(f"Mean IoU (mIoU) Globale: {class_ious.mean().item():.4f}")
    print("-" * 30)
    for name, iou in zip(class_names, class_ious):
        print(f"IoU {name:<15}: {iou.item():.4f}")

    # 7. VISUALIZZAZIONE DELLA CONFUSION MATRIX
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix.numpy(), annot=False, cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix Normalizzata")
    plt.xlabel("Classe Predetta")
    plt.ylabel("Classe Reale")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()

    # 8. VISUALIZZAZIONE DELLE PREDIZIONI (Ultimo e penultimo batch)
    _plot_predictions(almost_last_images, almost_last_masks, almost_last_preds)
    _plot_predictions(last_images, last_masks, last_preds, 2)


def _plot_predictions(images, true_masks, pred_masks, num_image = 0):
    """Denormalizza e plotta Immagine Reale | Maschera Reale | Predizione"""
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
    images = images * std + mean # Denormalizzazione per il plot
    
    batch_size = images.size(0)
    fig, axes = plt.subplots(batch_size, 3, figsize=(12, 4 * batch_size), squeeze=False)
    
    for i in range(batch_size):
        img_np = images[i].permute(1, 2, 0).cpu().numpy()
        true_mask_np = true_masks[i].cpu().numpy()
        pred_mask_np = pred_masks[i].cpu().numpy()
        
        # Clip per evitare fastidiosi warning di matplotlib sui pixel fuori range [0, 1]
        img_np = img_np.clip(0, 1)
        
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title("Immagine")
        axes[i, 0].axis("off")
        
        axes[i, 1].imshow(true_mask_np, cmap="nipy_spectral", vmin=0, vmax=20)
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")
        
        axes[i, 2].imshow(pred_mask_np, cmap="nipy_spectral", vmin=0, vmax=20)
        axes[i, 2].set_title("Predizione Rete")
        axes[i, 2].axis("off")

    plt.tight_layout()
    if num_image == 0:
        plt.savefig("predictions_sample.png")
    else:
        name_image = "predictions_sample" + str(num_image) + ".png"
        plt.savefig(name_image)
    plt.show()

if __name__ == "__main__":
    main()