import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
from glob import glob
from PIL import Image
import wandb

# ------------------ WANDB & GPU SETUP ------------------
os.environ["https_proxy"] = "http://hpc-proxy00.city.ac.uk:3128"
os.environ["WANDB_API_KEY"] = "7841fc270d1a6aa851fe30c369a2fe0851d31f55"

print("✅ Environment setup complete.")

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# ------------------ CONSTANTS ------------------
SEQUENCE_LENGTH = 16
IMG_SIZE = 224
BATCH_SIZE = 4
NUM_CLASSES = 8
CLASS_NAMES = ['Arrest','Arson','Assault','Burglary','Explosion','Fighting','NormalVideos','Shooting']
label_encoder = LabelEncoder()
label_encoder.fit(CLASS_NAMES)

# ------------------ DATASET CLASSES ------------------
class VideoDatasetMulti(Dataset):
    def __init__(self, base_dir, label_encoder, sequence_length=16, image_size=224, stride=4):
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.samples = []
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')

        for class_name in os.listdir(base_dir):
            class_path = os.path.join(base_dir, class_name)
            if not os.path.isdir(class_path): continue
            for video_folder in os.listdir(class_path):
                video_path = os.path.join(class_path, video_folder)
                frame_paths = sorted(glob(os.path.join(video_path, '*.png')))
                label = label_encoder.transform([class_name])[0]
                for i in range(0, len(frame_paths) - sequence_length + 1, stride):
                    clip = frame_paths[i:i+sequence_length]
                    if len(clip) == sequence_length:
                        self.samples.append((clip, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        paths, label = self.samples[idx]
        imgs = [Image.open(p).convert("RGB").resize((224, 224)) for p in paths]
        processed = self.processor(images=imgs, return_tensors="pt")
        return processed['pixel_values'].squeeze(0).permute(1, 0, 2, 3), torch.tensor(label)

class VideoDataset3D(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        pixel_values, label = self.base_dataset[idx]
        pixel_values = pixel_values.permute(1, 0, 2, 3)  # (C, T, H, W)
        return pixel_values, label

# ------------------ MODEL ------------------
class Finetuned3DCNNWithAttention(nn.Module):
    def __init__(self, num_classes=8, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 5, 5), stride=1, padding=(1, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),

            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),

            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),

            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),

            nn.Dropout3d(p=dropout) if dropout else nn.Identity(),
            nn.AdaptiveAvgPool3d((None, 1, 1))
        )

        self.attn = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(256)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)                # (B, 256, T, 1, 1)
        x = x.view(x.size(0), 256, -1)      # (B, 256, T)
        x = x.permute(0, 2, 1)              # (B, T, 256)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(attn_out + x)
        x = x.mean(dim=1)                   # (B, 256)
        return self.fc(x)

# ------------------ DATALOADERS ------------------
def create_dataset(path, multi=True):
    base_class = VideoDatasetMulti if multi else VideoDatasetSingle
    base_dataset = base_class(path, label_encoder, sequence_length=SEQUENCE_LENGTH, image_size=IMG_SIZE)
    return VideoDataset3D(base_dataset)

train_path = "/users/adhg808/sharedscratch/inm705_cw/data_trimmed/Train"
test_path = "/users/adhg808/sharedscratch/inm705_cw/data_trimmed/Test"

train_dataset = create_dataset(train_path, multi=True)
test_dataset = create_dataset(test_path, multi=True)

print(f"Train dataset loaded with {len(train_dataset)} samples.")
print(f"Test dataset loaded with {len(test_dataset)} samples.")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ------------------ TRAINING ------------------
def evaluate(model, loader, criterion, return_metrics=False):
    model.eval()
    total_loss = 0.0
    total_preds, total_probs, total_labels = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            print(f"[EVAL] Batch shape before permute: {x.shape}")
            x = x.permute(0, 2, 1, 3, 4)
            print(f"[EVAL] Batch shape after permute: {x.shape}")
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            loss = criterion(logits, y)

            total_loss += loss.item()
            total_probs.append(probs.cpu())
            total_preds.append(torch.argmax(probs, dim=1).cpu())
            total_labels.append(y.cpu())

    y_true = torch.cat(total_labels).numpy()
    y_pred = torch.cat(total_preds).numpy()
    y_probs = torch.cat(total_probs).numpy()

    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
    except ValueError:
        auc = 0.0

    avg_loss = total_loss / len(loader)
    if return_metrics:
        return avg_loss, acc, auc
    else:
        print(f"Loss: {avg_loss:.4f} | Accuracy: {acc:.4f} | AUC: {auc:.4f}")

# Set class weights
class_weights = torch.tensor([1.0]*NUM_CLASSES)  # Replace with actual weights if needed

model = Finetuned3DCNNWithAttention(num_classes=NUM_CLASSES, dropout=0.5).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE), label_smoothing=0.1)
optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.0)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
scaler = GradScaler()

# WandB
wandb.init(project="3dcnn-final-phase", name="finetuned3dcnn-attention")
wandb.config.update({
    "dropout": 0.5,
    "lr": 5e-5,
    "weight_decay": 0.0,
    "label_smoothing": 0.1,
    "epochs": 3,
    "batch_size": BATCH_SIZE
})

EPOCHS = 3
print("Starting training loop...")

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS} --------------------")
    model.train()
    running_loss = 0.0

    for batch_idx, (x, y) in enumerate(train_loader):
        print(f"Batch {batch_idx+1}/{len(train_loader)}")
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()

        with autocast():
            logits = model(x.permute(0, 2, 1, 3, 4))  # (B, T, C, H, W) → (B, C, T, H, W)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    print(f"Finished training epoch {epoch+1} → Avg Train Loss: {avg_train_loss:.4f}")

    # Evaluation
    print("Running evaluation on training set...")
    train_loss, train_acc, train_auc = evaluate(model, train_loader, criterion, return_metrics=True)
    
    print("Running evaluation on test set...")
    test_loss, test_acc, test_auc = evaluate(model, test_loader, criterion, return_metrics=True)

    # WandB logging
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "train_auc": train_auc,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_auc": test_auc,
        "lr": scheduler.get_last_lr()[0]
    })

    print(f"Epoch {epoch+1} Summary:")
    print(f"  Train → Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | AUC: {train_auc:.4f}")
    print(f"  Test  → Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | AUC: {test_auc:.4f}")
    
    scheduler.step()

print("\nTraining completed!")
wandb.finish()