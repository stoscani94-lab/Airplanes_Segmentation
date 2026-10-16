import lightning as L

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from dataset import SegmentationDataModule
from architecture import get_custom_baseline, get_unet, get_deeplabv3plus, get_transformer_seg
from lightning_module import SegmentationTask # Lo finiremo nello Step 4

def main():
    # 1. Configurazione Dati
   
    datamodule = SegmentationDataModule(data_dir="dataset/VOC_excercise/", batch_size=16)

    # 2. Scelta Architettura (Commenta/Decommenta per i vari esperimenti)
    # net = get_custom_baseline(num_classes=21)
    net = get_unet(num_classes=21)
    # net = get_deeplabv3plus(num_classes=21)
    # net = get_transformer_seg(num_classes=21)

    # 3. Inizializzazione Task
    task = SegmentationTask(model=net, lr=1e-3, num_classes=21)

    # 4. Configurazione Tracking
    # Sostituisci "nome-run" in base all'architettura per distinguerle sulla dashboard
    wandb_logger = WandbLogger(project="semantic-segmentation-benchmark", name="unet-resnet34")

    # 5. Configurazione Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="best-model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,         # Salva solo il modello con la loss migliore
        monitor="val_loss",
        mode="min"
    )
    
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,          # Se la val_loss non scende per 10 epoche, ferma il training
        mode="min"
    )

    # 6. Inizializzazione Trainer
    trainer = L.Trainer(
        max_epochs=50,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping],
        accelerator="auto",   # Trova automaticamente GPU (CUDA/MPS) o CPU
        devices="auto",
        fast_dev_run=True,  # <--- IL MAGIC SWITCH per fare test rapido
        overfit_batches=1,   # <--- Usa SOLO il primo batch e lo ripete all'infinito 
        log_every_n_steps=10
    )

    # 7. Avvio Addestramento
    trainer.fit(task, datamodule=datamodule)
    
    # 8. (Opzionale) Test automatico a fine training sul miglior modello salvato
    # trainer.test(task, datamodule=datamodule, ckpt_path="best")

if __name__ == "__main__":
    main()