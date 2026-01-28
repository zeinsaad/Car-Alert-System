# modules/preprocessing.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_transforms(img_size=64, train=True):
    if train:
        return transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((img_size, img_size)),
            transforms.RandomAffine(
                degrees=10,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        return transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

def load_dataset(dataset_path, img_size=64):
    dataset = datasets.ImageFolder(
        dataset_path,
        transform=get_transforms(img_size, train=True)
    )
    print("Classes found:", dataset.class_to_idx)
    
    return dataset

def create_dataloaders(dataset, batch_size=32, val_split=0.2, seed=42):
    total_size = len(dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator)

    train_ds.dataset.transform = get_transforms(train=True)
    val_ds.dataset.transform = get_transforms(train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")
    return train_loader, val_loader
