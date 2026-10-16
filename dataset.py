import os
import torch
import lightning as L
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import tv_tensors

# ==========================================
# 1. CLASSE DATASET (Livello Singola Immagine)
# ==========================================
class SegmentationDataset(Dataset):
    def __init__(self, img_paths, mask_paths):
        """
        Riceve solo le liste dei percorsi. 
        (esempio: img_paths = ["images/img_001.jpg", "images/img_002.jpg", ...]).
        Non fa augmentation qui, prepara solo i dati per il batching.
        """
        self.img_paths = img_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Lettura I/O nativa ultra-rapida (decodifica diretta in Tensor)
        img = read_image(self.img_paths[idx])
        mask = read_image(self.mask_paths[idx])

        # PyTorch per segmentazione vuole maschere di interi [H, W] (senza il canale)
        # read_image restituisce [1, H, W], quindi rimuoviamo la prima dimensione e castiamo.
        mask = mask.squeeze(0).to(torch.long)

        # Incapsuliamo nei tv_tensors per la futura augmentation con tensorvision v2
        img = tv_tensors.Image(img)
        mask = tv_tensors.Mask(mask)

        return img, mask


# ==========================================
# 2. CLASSE DATAMODULE (Livello Intero Dataset)
# ==========================================
class SegmentationDataModule(L.LightningDataModule):

    def __init__(self, data_dir: str, batch_size: int = 16, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    # Percorsi specifici della tua struttura
        self.images_dir = os.path.join(data_dir, "images")
        self.targets_dir = os.path.join(data_dir, "targets")
    
        # Inizializziamo i dataset a None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def setup(self, stage: str = None):
        """
        Legge i file .txt e costruisce le liste dei percorsi completi.
        """
        if stage == "fit" or stage is None:
            # Creiamo le liste per il Training
            train_imgs, train_masks = self._parse_split_file("train.txt")
            self.train_ds = SegmentationDataset(train_imgs, train_masks)
            
            # Creiamo le liste per la Validation
            val_imgs, val_masks = self._parse_split_file("val.txt")
            self.val_ds = SegmentationDataset(val_imgs, val_masks)

        if stage == "test" or stage is None:
            # Creiamo le liste per il Test
            test_imgs, test_masks = self._parse_split_file("test.txt")
            self.test_ds = SegmentationDataset(test_imgs, test_masks)

    def _parse_split_file(self, txt_filename: str):
        """
        Legge un file .txt e restituisce le liste dei percorsi assoluti 
        per le immagini e per le relative maschere.
        """
        txt_path = os.path.join(self.data_dir, txt_filename)
        
        with open(txt_path, 'r') as f:
            # Rimuoviamo eventuali spazi vuoti o ritorni a capo (\n)
            filenames = [line.strip() for line in f.readlines() if line.strip()]

        img_paths = []
        mask_paths = []
        
        for name in filenames:
            # ATTENZIONE: Se i file .txt non hanno l'estensione (es. dicono solo "img_001")
            # va aggiunta qui. Ad esempio f"{name}.jpg" e f"{name}.png".
            # Se invece nel txt c'è già scritto "img_001.jpg", lascia solo 'name'.
            
            # Nel txt ci sono i nomi senza estensione, quindi li aggiungiamo:
            img_paths.append(os.path.join(self.images_dir, f"{name}.jpg"))
            mask_paths.append(os.path.join(self.targets_dir, f"{name}.png"))
            
        return img_paths, mask_paths

    def prepare_data(self):
        """
        Chiamato SOLO UNA VOLTA su 1 singola GPU/CPU. 
        Qui ci metti logica di download o check file.
        (Es. eseguire il tuo script download_dataset.py se i dati mancano)
        """
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Dataset non trovato in {self.data_dir}. Esegui download_dataset.py")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=True,     # Fondamentale per spostare i dati su GPU velocemente
            drop_last=True       # Evita batch piccoli finali che possono dare noia a BatchNorm
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True
        )
        
    def _get_filepaths(self, directory):
        """Metodo helper per trovare immagini e maschere (da personalizzare in base al tuo dataset)"""
        # LOGICA CUSTOM: Restituisce due liste di percorsi ordinati (es. list(img_paths), list(mask_paths))
        pass


