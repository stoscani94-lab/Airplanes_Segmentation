import lightning as L

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.callbacks import RichProgressBar, RichModelSummary
from dataset import SegmentationDataModule
from architecture import get_custom_baseline, get_unet, get_deeplabv3plus, get_transformer_seg
from lightning_module import SegmentationTask # Lo finiremo nello Step 4

def main():
    # 1. Configurazione Dati
   # num_workers = 0 per CPU/mac, num_workers = 4 per GPU. Posso anche fare num_workers = num_gpus-1
    datamodule = SegmentationDataModule(data_dir="dataset/VOC_exercise/", batch_size=16, num_workers=2)

    # 2. Scelta Architettura (input interattivo)
    print("Scegli un numero da 1 a 4 per selezionare la rete: ")
    print("1) Custom Baseline")
    print("2) U-Net")
    print("3) Deeplab-v3")
    print("4) Transformer Segmentation")

    while True:
        try:
            choice = int(input("La tua scelta: "))
            if 1 <= choice <= 4:
                break
            print("Scelta non valida: inserisci un numero da 1 a 4.")
        except ValueError:
            print("Input non valido: inserisci un numero intero.")
    
    if choice == 1:
        net = get_custom_baseline(num_classes=21)
        nome_run = "Custom-Baseline"
    elif choice == 2:
        net = get_unet(num_classes=21)
        nome_run = "U-Net"
    elif choice == 3:
        net = get_deeplabv3plus(num_classes=21)
        nome_run = "DeepLab-v3"
    elif choice == 4:
        net = get_transformer_seg(num_classes=21)
        nome_run = "Transformer"
    else: 
        print("Something unexpected happened.")

    # 3. Inizializzazione Task
    task = SegmentationTask(model=net, lr=[1e-4, 1e-3], num_classes=21)

    # 4. Configurazione Tracking
    wandb_logger = WandbLogger(project="semantic-segmentation-benchmark", name=nome_run)

    # 5. Configurazione Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename=f"best-model-{nome_run}-{{epoch:02d}}-{{val-loss:.2f}}", 
        # correzioni qui non ho capito perchè vorrebbe che sia
        # filename=f"best-model-{nome_run}-{{epoch:02d}}-{{val/loss:.2f}}", 
        save_top_k=1,         # Salva solo il modello con la loss migliore
        monitor="val/loss",
        mode="min"
    )
    
    early_stopping = EarlyStopping(
        monitor="val/loss",
        patience=10,          # Se la val_loss non scende per 10 epoche, ferma il training
        mode="min"
    )

    # Inizializza i componenti Rich
    rich_progress_bar = RichProgressBar()
    rich_model_summary = RichModelSummary(max_depth=2) # Mostra i layer fino a profondità 2

    # 6. Inizializzazione Trainer
    trainer = L.Trainer(
        max_epochs=50,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping, rich_progress_bar, rich_model_summary],
        accelerator="auto",   # Trova automaticamente GPU (CUDA/MPS) o CPU
        devices="auto",
        # fast_dev_run=True,  # <--- IL MAGIC SWITCH per fare test rapido
        # overfit_batches=1,   # <--- Usa SOLO il primo batch e lo ripete all'infinito 
        log_every_n_steps=10
    )
    

    
    # 7. Avvio Addestramento
    trainer.fit(task, datamodule=datamodule)
    
    # 8. (Opzionale) Test automatico a fine training sul miglior modello salvato
    # trainer.test(task, datamodule=datamodule, ckpt_path="best")

if __name__ == "__main__":
    main()