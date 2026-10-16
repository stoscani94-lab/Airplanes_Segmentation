import torch
import torch.nn as nn
import lightning as L
from torchvision.transforms import v2
from torchmetrics.classification import MulticlassJaccardIndex
import wandb

class SegmentationTask(L.LightningModule):
    
    def __init__(self, model: nn.Module, num_classes: int = 21, lr: float = 1e-3):
        super().__init__()
        # Ignoriamo il salvataggio dell'architettura nei log per non appesantirli
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.num_classes = num_classes
        self.lr = lr
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Inizializzazione TorchMetrics: MulticlassJaccardIndex = mIoU
        # Creiamo istanze separate per train e val per evitare che i calcoli si mescolino
        self.train_iou = MulticlassJaccardIndex(num_classes=num_classes, average="macro")
        self.val_iou = MulticlassJaccardIndex(num_classes=num_classes, average="macro")
        
        # Pipeline di Augmentation (Vettorializzate su GPU)
        self.train_transforms = v2.Compose([
            v2.RandomResizedCrop(size=(256, 256), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.2, contrast=0.2),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transforms = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def on_after_batch_transfer(self, batch, dataloader_idx):
        x, y = batch
        if self.trainer.training:
            x, y = self.train_transforms(x, y)
        else:
            x, y = self.val_transforms(x, y)
        return x, y

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Estraiamo le classi predette (argmax sulle probabilità di output)
        preds = torch.argmax(logits, dim=1)
        
        # Log di Lightning: passando l'oggetto train_iou, lui accumula automaticamente per l'intera epoca
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/mIoU", self.train_iou(preds, y), on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mIoU", self.val_iou(preds, y), on_step=False, on_epoch=True, prog_bar=True)

        # W&B Visual Tracking: Logghiamo le maschere solo del primo batch di validazione
        if batch_idx == 0 and isinstance(self.logger, L.pytorch.loggers.WandbLogger):
            self._log_wandb_masks(x, y, preds)
            
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def _log_wandb_masks(self, x, y, preds, num_samples=4):
        """Helper per loggare le predizioni sovrapposte in W&B."""
        # De-normalizziamo le immagini per la corretta visualizzazione in RGB
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        imgs = x * std + mean
        
        n = min(x.size(0), num_samples)
        mask_list = []
        
        for i in range(n):
            # Convertiamo in NumPy (W&B richiede array standard)
            img_np = imgs[i].permute(1, 2, 0).cpu().numpy()
            true_mask = y[i].cpu().numpy()
            pred_mask = preds[i].cpu().numpy()
            
            # Formato nativo di W&B per le maschere interattive
            mask_list.append(wandb.Image(img_np, masks={
                "predictions": {"mask_data": pred_mask},
                "ground_truth": {"mask_data": true_mask}
            }))
            
        # Log diretto nell'esperimento
        self.logger.experiment.log({"val/predictions": mask_list})