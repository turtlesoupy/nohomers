from .torch_datasets import SimpleVisionExample, SimpleVisionDataset, split_train_valid_test, pil_loader
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import pandas as pd
import numpy as np
import torch
import torchvision
import pydash as py_
from tqdm.auto import tqdm
from uuid import uuid4
from PIL import Image
from pathlib import Path
import json
import copy
from multiprocessing.pool import ThreadPool
from typing import List
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics

_input_size = 224
_train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(p=0.2),
    torchvision.transforms.RandomAffine(
        degrees=10,
        scale=(0.9, 0.95),
        shear=(5, 5),
        translate=(0.05, 0.05),
        fillcolor=(255, 255, 255),
        resample=Image.BICUBIC,
    ),
    (lambda x: x.resize(size=(_input_size, _input_size), resample=Image.BICUBIC)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_eval_transforms = torchvision.transforms.Compose([
    (lambda x: x.resize(size=(_input_size, _input_size), resample=Image.BICUBIC)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def make_network_input_from_images(images: List[Image.Image]) -> torch.Tensor:
    image_tensors = [_eval_transforms(e) for e in images]
    batch = default_collate(image_tensors)
    return batch


def make_train_test_datasets(examples: List[SimpleVisionExample]):
    train_dataset, _, test_dataset = [
        SimpleVisionDataset(
            e, 
            transform=(_train_transforms if i == 0 else _eval_transforms), 
        ) 
        for i, e in enumerate(split_train_valid_test(
            examples, train_size=0.8, valid_size=0, test_size=0.2
        ))
    ]

    return train_dataset, test_dataset


class VisionOnlyCleaner(nn.Module):
    def __init__(self, latent_dim=None):
        super().__init__()
        self.model_ft = torchvision.models.resnet50(pretrained=True)
        for param in self.model_ft.parameters():
            param.requires_grad = False
        num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = torch.nn.Linear(num_ftrs, 2)

    def forward(self, x, latent):
        return self.model_ft.forward(x)


class VisionLatentCleaner(nn.Module):
    def __init__(self, latent_dim, freeze_resnet=True):
        super().__init__()
        self.model_ft = torchvision.models.resnet50(pretrained=True)
        if freeze_resnet:
            for param in self.model_ft.parameters():
                param.requires_grad = False

        num_ft_features = self.model_ft.fc.in_features
        self.model_ft.fc = torch.nn.Identity()

        self.extended_model = torch.nn.Sequential(
            torch.nn.Linear(num_ft_features + latent_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 2),
        )

    def forward(self, x, latent):
        resnet_out = self.model_ft.forward(x)
        to_input = torch.hstack((resnet_out, latent))
        return self.extended_model(to_input)

def train_cleaner(model, train_dataset, eval_dataset, batch_size=50, num_epochs=5, device="cuda:0", lr=0.001, l2_reg=0.005, clip_norm=0.5, workers=10):
    optimizer_ft = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)
    train_dataloader = DataLoader(
        train_dataset, num_workers=workers, batch_size=batch_size, pin_memory=True, collate_fn=train_dataset.collate_fn,
    )
    eval_dataloader = DataLoader(
        eval_dataset, num_workers=workers, batch_size=batch_size, pin_memory=True, collate_fn=eval_dataset.collate_fn,
    )

    best_epoch = None
    best_eval_auc = None
    best_eval_state = None

    def forward(batch):
        x, latents, labels = batch
        x = x.to(device)
        latents = latents.to(device)
        labels = labels.to(device)
        ret = model(x, latents)
        loss = torch.nn.CrossEntropyLoss()(ret, labels)
        with torch.no_grad():
            probs = torch.nn.Softmax(dim=1)(ret)[:, 1]
        
        return loss, probs

    for epoch_num in tqdm(range(num_epochs), "Epoch"):
        optimizer_ft.zero_grad()
        train_loss = 0
        train_size = 0
        train_labs = []
        train_probs = []
        for i, batch in tqdm(enumerate(train_dataloader), "Train Batch"):
            train_size += batch[0].size(0)
            model.train()
            loss, batch_probs = forward(batch)
            train_probs.extend(batch_probs.cpu().numpy().tolist())
            train_labs.extend(batch[-1].cpu().numpy().tolist())
            loss.backward()
            train_loss += loss

            if clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer_ft.step()
            optimizer_ft.zero_grad()
        
        train_loss /= train_size
        fpr, tpr, _ = metrics.roc_curve(train_labs, train_probs, pos_label=1)
        train_auc = metrics.auc(fpr, tpr)

        with torch.no_grad():
            model.eval()
            eval_loss = 0
            eval_size = 0

            eval_labs = []
            eval_probs = []

            for i, batch in enumerate(eval_dataloader):
                eval_size += batch[0].size(0)
                batch_loss, batch_probs = forward(batch)
                eval_probs.extend(batch_probs.cpu().numpy().tolist())
                eval_labs.extend(batch[-1].cpu().numpy().tolist())
                eval_loss += batch_loss

            eval_loss /= eval_size

            fpr, tpr, _ = metrics.roc_curve(eval_labs, eval_probs, pos_label=1)
            eval_auc = metrics.auc(fpr, tpr)
                
            if best_eval_auc is None or eval_auc > best_eval_auc:
                best_eval_auc = eval_auc
                best_eval_state = copy.copy(model.state_dict())
                best_epoch = epoch_num
        
        print(f"Epoch {epoch_num}: train_loss={train_loss:.3f}, train_auc={train_auc:.3f}, eval_loss={eval_loss:.3f}, eval_auc={eval_auc:.3f}")
    
    print(f"Loading best eval auc {best_eval_auc} from epoch {best_epoch}")
    model.load_state_dict(best_eval_state)


@torch.no_grad()
def scores_for_images(cleaner, images, latents):
    batch = make_network_input_from_images(images).cuda()
    latent_batch = torch.vstack(latents).cuda()
    batch_scores = cleaner.forward(batch, latent_batch)
    probs = torch.nn.Softmax(dim=1)(batch_scores)
    return [p[1].item() for p in probs]

def load_cleaner(path, device="cuda:0", klass=VisionLatentCleaner, latent_dim=None):
    model = klass(latent_dim=latent_dim)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model.to(device)

def save_cleaner(model, path):
    torch.save(model.state_dict(), path)
