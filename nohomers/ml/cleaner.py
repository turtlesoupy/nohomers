from .torch_datasets import SimpleVisionExample, SimpleVisionDataset, split_train_valid_test, pil_loader
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import pandas as pd
import numpy as np
import torch
import torchvision
import pydash as py_
from tqdm import tqdm
from uuid import uuid4
from PIL import Image
from pathlib import Path
import json
import copy
from multiprocessing.pool import ThreadPool
from typing import List

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


def train_cleaner(train_dataset, eval_dataset, batch_size=50, num_epochs=5, device="cuda:0", lr=0.001, l2_reg=0.005, clip_norm=0.5, workers=10):
    model_ft = torchvision.models.resnet50(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = torch.nn.Linear(num_ftrs, 2)
    model_ft = model_ft.to(device)

    optimizer_ft = torch.optim.Adam(model_ft.fc.parameters(), lr=lr, weight_decay=l2_reg)
    train_dataloader = DataLoader(
        train_dataset, num_workers=workers, batch_size=batch_size, pin_memory=True, collate_fn=train_dataset.collate_fn,
    )
    eval_dataloader = DataLoader(
        eval_dataset, num_workers=workers, batch_size=batch_size, pin_memory=True, collate_fn=eval_dataset.collate_fn,
    )

    best_epoch = None
    best_eval_loss = None
    best_eval_state = None

    def forward(batch):
        batch, labels = batch
        batch = batch.to(device)
        labels = labels.to(device)
        ret = model_ft(batch)
        loss = torch.nn.CrossEntropyLoss()(ret, labels)
        
        return loss

    for epoch_num in tqdm(range(num_epochs), "Epoch"):
        optimizer_ft.zero_grad()
        train_loss = 0
        train_size = 0
        for i, batch in tqdm(enumerate(train_dataloader), "Train Batch"):
            train_size += batch[0].size(0)
            model_ft.train()
            loss = forward(batch)
            loss.backward()
            train_loss += loss

            if clip_norm:
                torch.nn.utils.clip_grad_norm_(model_ft.parameters(), clip_norm)
            optimizer_ft.step()
            optimizer_ft.zero_grad()
        
        train_loss /= train_size

        with torch.no_grad():
            model_ft.eval()
            eval_loss = 0
            eval_size = 0
            for i, batch in enumerate(eval_dataloader):
                eval_size += batch[0].size(0)
                eval_loss += forward(batch)

            eval_loss /= eval_size
                
            if best_eval_loss is None or eval_loss < best_eval_loss:
                best_eval_loss = eval_loss 
                best_eval_state = copy.copy(model_ft.state_dict())
                best_epoch = epoch_num
        
        print(f"Epoch {epoch_num}: train_loss={train_loss:.3f}, eval_loss={eval_loss:.3f}")
    
    print(f"Loading best eval {best_eval_loss} from epoch {best_epoch}")
    model_ft.load_state_dict(best_eval_state)

    return model_ft

def scores_for_images(cleaner, images):
    batch = make_network_input_from_images(images).cuda()
    batch_scores = cleaner.forward(batch)
    probs = torch.nn.Softmax(dim=1)(batch_scores)
    return [p[1].item() for p in probs]

def load_cleaner(path, device="cuda:0"):
    model = torchvision.models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model.to(device)

def save_cleaner(model, path):
    torch.save(model.state_dict(), path)
