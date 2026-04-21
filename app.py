import os


from google.colab import drive
drive.mount('/content/drive')

print("\n--- Listing contents of MyDrive ---")
!ls "/content/drive/MyDrive/"
print("\n--- Listing contents of Tooth_DL if it exists ---")
!ls "/content/drive/MyDrive/Tooth_DL/"
# --- ADDED CODE END ---

DATASET_PATH = "/content/drive/MyDrive/Tooth_DL"


import os, cv2, copy, torch, random, numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# Preprocessing

def resize_with_padding(image, target=224):
    h, w = image.shape[:2]
    scale = target / max(h, w)
    nh, nw = int(h * scale), int(w * scale)

    image = cv2.resize(image, (nw, nh))
    pad_h, pad_w = target - nh, target - nw

    return cv2.copyMakeBorder(
        image,
        pad_h//2, pad_h-pad_h//2,
        pad_w//2, pad_w-pad_w//2,
        cv2.BORDER_CONSTANT,
        value=0
    )


class DentalDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform
        self.clahe = cv2.createCLAHE(2.0, (8,8))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx])

        if img is None:
            img = np.zeros((224,224,3), dtype=np.uint8)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = resize_with_padding(gray)

        gray = cv2.bilateralFilter(gray, 7, 50, 50)
        gray = self.clahe.apply(gray)
        gray = np.uint8(np.power(gray / 255.0, 0.8) * 255)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        img = Image.fromarray(rgb)

        if self.transform:
            img = self.transform(img)

        return img, self.labels[idx]


image_paths, labels = [], []

class_names = sorted([
    d for d in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, d))
])

valid_ext = ('.jpg', '.jpeg', '.png', '.bmp')

for idx, cls in enumerate(class_names):
    folder = os.path.join(DATASET_PATH, cls)
    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        if os.path.isfile(path) and f.lower().endswith(valid_ext):
            image_paths.append(path)
            labels.append(idx)

image_paths = np.array(image_paths)
labels = np.array(labels)

print("Classes:", class_names)
print("Total Images:", len(image_paths))

#  Data Splitting
X_train, X_temp, y_train, y_temp = train_test_split(
    image_paths, labels,
    test_size=0.30,
    stratify=labels,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42
)


MEAN = [0.35]*3
STD = [0.31]*3

train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

val_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

train_loader = DataLoader(
    DentalDataset(X_train, y_train, train_tf),
    batch_size=16,
    shuffle=True
)

val_loader = DataLoader(
    DentalDataset(X_val, y_val, val_tf),
    batch_size=16,
    shuffle=False
)

test_loader = DataLoader(
    DentalDataset(X_test, y_test, val_tf),
    batch_size=16,
    shuffle=False
)

# FOCAL LOSS

class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        logp = F.log_softmax(inputs, dim=1)

        if targets.ndim == 2:
            ce = -(targets * logp).sum(dim=1)
            alpha_factor = (targets * self.alpha.unsqueeze(0)).sum(dim=1)
        else:
            ce = F.cross_entropy(inputs, targets, reduction='none')
            alpha_factor = self.alpha[targets]

        pt = torch.exp(-ce)
        loss = alpha_factor * ((1 - pt) ** self.gamma) * ce

        return loss.mean()

# CBAM
class CBAM(nn.Module):
    def __init__(self, ch):
        super().__init__()

        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch//16, 1),
            nn.ReLU(),
            nn.Conv2d(ch//16, ch, 1),
            nn.Sigmoid()
        )

        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x * self.ca(x)

        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)

        x = x * self.sa(torch.cat([avg, mx], dim=1))
        return x


class DenseNet121_CBAM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        base = models.densenet121(
            weights=models.DenseNet121_Weights.DEFAULT
        )

        self.features = base.features
        self.cbam = CBAM(1024)
        self.pool = nn.AdaptiveAvgPool2d((1,1))

        self.classifier = nn.Sequential(
            nn.Linear(1024,256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = self.cbam(x)
        x = self.pool(x)
        x = torch.flatten(x,1)
        return self.classifier(x)

    def freeze_early(self):
        for name, param in self.features.named_parameters():
            if "denseblock3" not in name and "denseblock4" not in name:
                param.requires_grad = False

model = DenseNet121_CBAM(len(class_names)).to(device)


def mixup(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)

    idx = torch.randperm(x.size(0)).to(x.device)

    y_one = F.one_hot(y, len(class_names)).float()
    y_mix = lam*y_one + (1-lam)*y_one[idx]

    x_mix = lam*x + (1-lam)*x[idx]

    return x_mix, y_mix


weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

weights = torch.tensor(weights, dtype=torch.float).to(device)

criterion = FocalLoss(weights)
optimizer = optim.AdamW(model.parameters(), lr=3e-4)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    patience=3
)

EPOCHS = 40
FREEZE_EPOCH = 15
PATIENCE = 5

best_acc = 0
best_wts = copy.deepcopy(model.state_dict())
es = 0


for epoch in range(EPOCHS):

    if epoch == FREEZE_EPOCH:
        model.freeze_early()
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=5e-5
        )

    model.train()
    correct, total = 0, 0

    for x, y in tqdm(train_loader):
        x, y = x.to(device), y.to(device)

        x_mix, y_mix = mixup(x, y)

        optimizer.zero_grad()
        out = model(x_mix)
        loss = criterion(out, y_mix)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    train_acc = correct / total

    # VALIDATION
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)

            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    val_acc = correct / total
    scheduler.step(val_acc)

    print(f"Epoch {epoch+1}: Train={train_acc:.4f} Val={val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        best_wts = copy.deepcopy(model.state_dict())
        es = 0
    else:
        es += 1
        if es >= PATIENCE:
            print("Early Stopping Triggered")
            break


# Test

model.load_state_dict(best_wts)
model.eval()

all_preds, all_labels, all_probs = [], [], []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)

        out = model(x)
        probs = torch.softmax(out, dim=1)

        all_preds.extend(out.argmax(1).cpu().numpy())
        all_labels.extend(y.numpy())
        all_probs.append(probs.cpu().numpy())

all_probs = np.concatenate(all_probs)


# Classification Report

print(classification_report(
    all_labels,
    all_preds,
    target_names=class_names
))


# CONFUSION MATRIX

cm = confusion_matrix(all_labels, all_preds)

sns.heatmap(cm, annot=True, fmt='d')
plt.show()


y_bin = label_binarize(
    all_labels,
    classes=list(range(len(class_names)))
)

plt.figure(figsize=(8,6))

for i in range(len(class_names)):
    fpr, tpr, _ = roc_curve(y_bin[:,i], all_probs[:,i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{class_names[i]} AUC={roc_auc:.3f}")

plt.plot([0,1],[0,1],'k--')
plt.legend()
plt.title("ROC Curve")
plt.show()