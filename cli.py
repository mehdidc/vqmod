from clize import run
from collections import Counter
import numpy as np

import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.optim import Adam
import torch.nn.functional as F

from models import Model
from models import ModularNet

from data import SubSample

def train(*, 
        train_folder='data/mnist/train', 
        valid_folder='data/mnist/valid', 
        epochs=100,
        device='cpu', 
        num_workers=4,
        log_interval=10,
        image_size=32):
    batch_size = 32
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = ImageFolder(
        train_folder,
        transform=transform
    )
    valid_dataset = ImageFolder(
        valid_folder,
        transform=transform
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers
    )

    model = Model()
    model = model.to(device)
    model.transform = transform
    optim = Adam(model.parameters(), lr=0.001)
    niter = 0
    for epoch in range(epochs):
        for X, y in train_dataloader:
            X = X.to(device)
            y = y.to(device)
            y_pred, ctl, ctl_nearest, out_idx = model(X)
            model.zero_grad()
            vq_loss = ModularNet.vq_loss_function(ctl, ctl_nearest)
            ce_loss = F.cross_entropy(y_pred, y)
            loss = ce_loss + vq_loss
            loss.backward()
            optim.step()
            _, y_pred_class = y_pred.max(dim=1)
            acc = (y_pred_class == y).float().mean()
            if niter % log_interval == 0:
                out_idx = out_idx.cpu().flatten().detach()
                prob = torch.zeros(len(model.modular.components))
                for val in range(len(model.modular.components)):
                    prob[val] = (out_idx==val).float().mean() + 1e-7
                ent = (-prob * np.log(prob)).mean()
                print(f'niter: {niter:05d} loss: {loss.item():.2f}, ce: {ce_loss.item():.2f} vq: {vq_loss.item():.2f} ent: {ent.item():.2f} acc: {acc.item():.2f}')
                torch.save(model, 'model.th')
            niter += 1
        for loader, name in [ (train_dataloader, 'train'), (valid_dataloader, 'valid') ]:
            accs = []
            ce_losses = []
            for X, y in loader:
                X = X.to(device)
                y = y.to(device)
                y_pred, _, _, _ = model(X)
                _, y_pred_class = y_pred.max(dim=1)
                ce_loss = F.cross_entropy(y_pred, y)
                acc = (y_pred_class == y).float().mean()
                accs.append(acc.item())
                ce_losses.append(ce_loss.item())
            print(f'epoch {epoch} {name} acc: {np.mean(accs)} ce_loss: {np.mean(ce_losses)}')



if __name__ == '__main__':
    run([train])
