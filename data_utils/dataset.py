from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from pathlib import Path
from PIL import Image


class UBCOCEANDataset(Dataset):
    def __init__(self, root: str, infocsv: pd.DataFrame, thumnail: bool=True, type: str = "train", num_class: int = 5) -> None:
        self.type = type
        if type != "test":
            self.data_dir = Path(root, "train_thumbnails")
        else:
            self.data_dir = Path(root, "test_thumbnails")
        self.info = infocsv
        self.label_encoder = {
            "HGSC": 0,
            "EC": 1,
            "CC": 2,
            "LGSC": 3,
            "MC": 4
        }
        self.transform = transforms.Compose([
            transforms.Resize(size=(512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self) -> int:
        return len(self.info)
    
    def __getitem__(self, index) -> tuple:
        datapiece = self.info.iloc[index]
        label = self.label_encoder[datapiece["label"]]
        image = Image.open(Path(self.data_dir, f"{datapiece['image_id']}_thumbnail.png"))
        image = self.transform(image)
        return image, label


def get_dataloader(dataset, is_train, batchsize, num_worker):
    if is_train:
        dataloader = DataLoader(dataset, shuffle=True, batch_size=batchsize, num_workers=num_worker)
    else:
        dataloader = DataLoader(dataset, shuffle=False, batch_size=1, num_workers=num_worker)
    
    return dataloader
