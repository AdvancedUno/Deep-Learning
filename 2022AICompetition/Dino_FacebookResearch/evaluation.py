import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


def compute_knn(backbone, data_loader_train, data_loader_val):

    device = next(backbone.parameters()).device

    data_loaders = {
        "train": data_loader_train,
        "val": data_loader_val,
    }
    lists = {
        "X_train": [],
        "y_train": [],
        "X_val": [],
        "y_val": [],
    }

    for name, data_loader in data_loaders.items():
        for imgs, y in data_loader:
            imgs = imgs.to(device)
            lists[f"X_{name}"].append(backbone(imgs).detach().cpu().numpy())
            lists[f"y_{name}"].append(y.detach().cpu().numpy())

    arrays = {k: np.concatenate(l) for k, l in lists.items()}

    estimator = KNeighborsClassifier()
    estimator.fit(arrays["X_train"], arrays["y_train"])
    y_val_pred = estimator.predict(arrays["X_val"])

    acc = accuracy_score(arrays["y_val"], y_val_pred)

    return acc

def compute_embedding(backbone, data_loader):

    device = next(backbone.parameters()).device

    embs_l = []
    imgs_l = []
    labels = []

    for img, y in data_loader:
        img = img.to(device)
        embs_l.append(backbone(img).detach().cpu())
        imgs_l.append(((img * 0.224) + 0.45).cpu())  # undo norm
        labels.extend([data_loader.dataset.classes[i] for i in y.tolist()])

    embs = torch.cat(embs_l, dim=0)
    imgs = torch.cat(imgs_l, dim=0)

    return embs, imgs, labels