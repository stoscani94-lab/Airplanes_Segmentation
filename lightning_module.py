import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import lightning as L
from torchvision.transforms import v2
from torchmetrics.classification import MulticlassJaccardIndex
import wandb
from typing import Union, List, Tuple

# Loss function combinazione di Dice + CrossEntropy, con weighted classes per la CE
class CEDiceLoss(nn.Module):
    def __init__(self, ce_weights=None, dice_weight=0.5, ce_weight_multiplier=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight_multiplier = ce_weight_multiplier
        
        # La tua CE pesata attuale
        self.ce = nn.CrossEntropyLoss(weight=ce_weights)
        
        # La Dice Loss di smp configurata per il multiclasse
        # Da notare: smp prende i logits in ingresso (come la CE), quindi non serve fare la softmax prima!
        self.dice = smp.losses.DiceLoss(
            mode='multiclass'
        )
        
    def forward(self, logits, targets):
        loss_ce = self.ce(logits, targets)
        loss_dice = self.dice(logits, targets)
        return (self.ce_weight_multiplier * loss_ce) + (self.dice_weight * loss_dice)

# Loss function combinazione di Dice Loss + Focal Loss
class FocalDiceLoss(nn.Module):
    def __init__(self, dice_weight=0.5, focal_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        # La Focal Loss si bilancia da sola in base alla confidenza (gamma=2.0 di default in smp)
        self.focal = smp.losses.FocalLoss(mode='multiclass')
        self.dice = smp.losses.DiceLoss(mode='multiclass')
        
    def forward(self, logits, targets):
        loss_focal = self.focal(logits, targets)
        loss_dice = self.dice(logits, targets)
        return (self.focal_weight * loss_focal) + (self.dice_weight * loss_dice)

class SegmentationTask(L.LightningModule):
    
    def __init__(self, model: nn.Module, 
                    num_classes: int = 21, 
                    lr: Union[float, List[float], Tuple[float, float]] = 1e-3,
                    num_epochs: int = 50    
                ):

        super().__init__()
        # Ignoriamo il salvataggio dell'architettura nei log per non appesantirli
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.num_classes = num_classes
        self.lr = lr
        self.num_epochs = num_epochs
        
        # Pesi per la CrossEntropy
        # 1. Converti la lista in un Tensor PyTorch di tipo float
        # 2. Sposta il Tensor sullo stesso device del modello/dati
        # weights_list = [0.04,  1.19,  1.64,  1.17,  1.37,  1.28,  0.79,  0.89,  0.64,  1.09,  1.07,  1.00,  0.78,  1.09,  1.05,  0.38,  1.37,  1.20,  0.91, 0.83,  1.23]  # 1 classe background + 20 classi target
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # weights = torch.tensor(weights_list, dtype=torch.float32).to(device)

        # Choose 1 of the 3 loss: 
        # self.criterion = nn.CrossEntropyLoss(weight=weights)
        # self.criterion = CEDiceLoss(ce_weights=weights)
        self.criterion = FocalDiceLoss()
        
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

    # modifica di visualizzazione in WandB per avere le epoch sull'asse x e non gli step.
    def on_train_start(self):
        if isinstance(self.logger, L.pytorch.loggers.WandbLogger):
            self.logger.experiment.define_metric("epoch")
            self.logger.experiment.define_metric("train/*", step_metric="epoch")
            self.logger.experiment.define_metric("val/*", step_metric="epoch")

    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Estraiamo le classi predette (argmax sulle probabilità di output)
        preds = torch.argmax(logits, dim=1)
        
        # Log di Lightning: passando l'oggetto train_iou, lui accumula automaticamente per l'intera epoca
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
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

        # Controlliamo se lr è una tupla/lista di 2 elementi E se il modello ha un encoder separato (tipico di smp)
        if isinstance(self.lr, (list, tuple)) and len(self.lr) == 2 and hasattr(self.model, 'encoder'):
            # NOTA: metto decoder per primo, quello con il lr più alto, così se metto 2 lr ma il modello non 
            # ha encoder/decoder terrò solo il primo valore come valore per tutta la rete.
            decoder_lr, encoder_lr  = self.lr
            param_groups = [
                {"params": self.model.encoder.parameters(), "lr": encoder_lr},
                {"params": self.model.decoder.parameters(), "lr": decoder_lr},
                {"params": self.model.segmentation_head.parameters(), "lr": decoder_lr},
            ]
        else:
            # Fallback: LR unico per tutto
            # Gestiamo il caso in cui l'utente passi per sbaglio una lista di 1 elemento o il modello non sia smp
            single_lr = self.lr if isinstance(self.lr, float) else self.lr[0]
            param_groups = [{"params": self.model.parameters(), "lr": single_lr}]

        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs)

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
        self.logger.experiment.log({
            "val/predictions": mask_list,
            "epoch": self.current_epoch
        })