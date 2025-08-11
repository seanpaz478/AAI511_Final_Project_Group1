import pretty_midi
import os
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, classification_report, accuracy_score
from torch.amp import autocast, GradScaler

# Creating function for converting MIDI file to a piano roll representation
def midi_to_pianoroll(filepath, fs=100, cut_notes=False):
    try:
        midi = pretty_midi.PrettyMIDI(filepath)
        # Normalizing piano roll
        piano_roll = midi.get_piano_roll(fs=fs) / 127.0
        # Creating option of removing extremely high/low notes since they could potentially be noise
        if cut_notes:
            piano_roll = piano_roll[21:109, :]
        return piano_roll.astype(np.float32)
    except Exception as e:
        raise ValueError(f"Failed to process MIDI file '{filepath}': {e}")

# Simple CNN for testing PrettyMIDI
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)  # Shape: (batch, channels, 1, 1)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Creating cache builder script to reduce training time    
def preprocess_and_cache(root_dir, out_dir, fs=100, max_length=1000, allowed_composers=None, cut_notes=False):
    if allowed_composers is None:
        allowed_composers = ["Bach", "Chopin", "Mozart", "Beethoven"]

    os.makedirs(out_dir, exist_ok=True)
    label_map = {composer: idx for idx, composer in enumerate(sorted(allowed_composers))}

    for composer in allowed_composers:
        composer_dir = os.path.join(root_dir, composer)
        if not os.path.isdir(composer_dir):
            continue

        for root, _, files in os.walk(composer_dir):
            for file in files:
                if not file.lower().endswith((".mid", ".midi")):
                    continue

                filepath = os.path.join(root, file)
                base_filename = f"{composer}_{file}".replace(" ", "_").replace("/", "_")
                save_path = os.path.join(out_dir, f"{base_filename}.pt")

                if os.path.exists(save_path):
                    print(f"[SKIP] {file} already processed.")
                    continue

                try:
                    roll = midi_to_pianoroll(filepath, fs=fs, cut_notes=cut_notes)
                    if roll.shape[1] > max_length:
                        roll = roll[:, :max_length]
                    else:
                        pad = max_length - roll.shape[1]
                        roll = np.pad(roll, ((0, 0), (0, pad)), mode="constant")

                    tensor = torch.tensor(roll).unsqueeze(0)
                    label = label_map[composer]

                    torch.save((tensor, label), save_path)
                    print(f"[SAVED] {file} -> {save_path}")
                except Exception as e:
                    print(f"[ERROR] {filepath} skipped. Reason: {e}")
                    
def pitch_shift(x, shift_range=(-2, 2)):
    shift = np.random.randint(shift_range[0], shift_range[1] + 1)
    if shift == 0:
        return x
    x_shifted = x.clone()
    x_shifted[:, 0] = torch.clamp(x[:, 0] + shift, min=0, max=127)
    return x_shifted

def time_stretch(x, min_rate=0.9, max_rate=1.1):
    rate = np.random.uniform(min_rate, max_rate)

    while x.dim() > 2 and x.shape[0] == 1:
        x = x.squeeze(0)

    if x.dim() != 2:
        raise ValueError(f"Input to time_stretch must be 2D (channels, length). Got shape: {x.shape}")

    orig_length = x.shape[1]
    new_length = max(1, int(orig_length / rate))

    x_unsqueezed = x.unsqueeze(0)
    stretched = F.interpolate(x_unsqueezed, size=new_length, mode='linear', align_corners=False)
    stretched = stretched.squeeze(0) 

    return stretched

def pad_or_truncate(x, target_length=1000):
    length = x.shape[-1]
    if length == target_length:
        return x
    elif length > target_length:
        return x[..., :target_length]
    else:
        pad_size = target_length - length
        padding = torch.zeros(x.shape[:-1] + (pad_size,), dtype=x.dtype)
        return torch.cat([x, padding], dim=-1)
    
def velocity_jitter(x, jitter_range=(-10, 10)):
    x_jittered = x.clone()
    jitter = np.random.randint(jitter_range[0], jitter_range[1] + 1)
    x_jittered[:, 1] = torch.clamp(x_jittered[:, 1] + jitter, min=0, max=127)
    return x_jittered

class CachedMidiDataset(Dataset):
    def __init__(self, cache_dir, augment=False, class_balance=None, max_prob=0.8, min_prob=0.2):
        self.files = [os.path.join(cache_dir, f) for f in os.listdir(cache_dir) if f.endswith(".pt")]
        self._labels = [torch.load(f)[1] for f in self.files]
        self.augment = augment
        self.max_prob = max_prob
        self.min_prob = min_prob
        self.current_epoch = 0
        self.total_epochs = 1  

        if class_balance is None:
            from collections import Counter
            class_balance = Counter(self._labels)

        counts = np.array(list(class_balance.values()), dtype=np.float32)
        counts = counts / counts.max() 

        self.base_probs = {}
        for cls, norm_count in zip(class_balance.keys(), counts):
            rarity_factor = 1 - norm_count
            prob = min_prob + rarity_factor * (max_prob - min_prob)
            self.base_probs[cls] = prob

    def set_epoch_info(self, current_epoch, total_epochs):
        self.current_epoch = current_epoch
        self.total_epochs = total_epochs

    def get_decay_factor(self):
        progress = self.current_epoch / max(1, self.total_epochs - 1)
        return 1.0 - 0.5 * progress 

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x, y = torch.load(self.files[idx])

        if self.augment:
            aug_prob = self.base_probs[y] * self.get_decay_factor()
            if np.random.rand() < aug_prob:
                x = self.apply_augmentation(x)

        return x, y

    @property
    def labels(self):
        return self._labels
    
    def apply_augmentation(self, x):
        if np.random.rand() < 0.5:
            x = pitch_shift(x)
        if np.random.rand() < 0.5:
            x = velocity_jitter(x)
        return x 

# Function for creating more balanced train/val datasets
def stratified_split(labels, test_size=0.2, random_state=0):
    train_idx, val_idx = train_test_split(
        range(len(labels)),
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )
    return train_idx, val_idx

# Function for addressing class imbalance in dataset
def make_weighted_sampler_and_loss(dataset_subset, device):
    labels = [dataset_subset.dataset.labels[i] for i in dataset_subset.indices]
    class_counts = np.bincount(labels)
    
    # Class weights for sampler
    class_weights = (1.0 / class_counts) ** 0.5
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # Class weights for loss
    loss_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=loss_weights)

    return sampler, criterion

# Updating model to be a bi-directional LSTM with dropout and batch normalization (First model attempt)
class CNNBiLSTMClassifier(nn.Module):
    def __init__(self, num_classes=4, dropout_p=0.3):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )

        self.lstm = nn.LSTM(
            input_size=128 * 11,  
            hidden_size=64,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout_p
        )

        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(64 * 2, num_classes)

    def forward(self, x):
        x = self.cnn(x)  
        x = x.permute(0, 3, 1, 2)  
        b, t, c, f = x.shape
        x = x.reshape(b, t, c * f)  

        x, _ = self.lstm(x)
        x = x[:, -1, :]  

        x = self.dropout(x)
        x = self.fc(x)
        return x
    
# Creating early stopping class
class EarlyStopping:
    def __init__(self, patience=3, delta=0.0, mode='min'):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
        self.mode = mode

    def __call__(self, current_score, model):
        if self.best_score is None:
            self.best_score = current_score
            self.best_model_state = model.state_dict()
            self.counter = 0
        else:
            if (self.mode == 'min' and current_score < self.best_score - self.delta) or \
               (self.mode == 'max' and current_score > self.best_score + self.delta):
                self.best_score = current_score
                self.best_model_state = model.state_dict()
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True

    def step(self, current_score, model):
        self.__call__(current_score, model)

    def restore_best_weights(self, model):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            
# Creating model training/evaluation function for use with multiple models
def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, device,
                       num_epochs=20, early_stopping=None, use_amp=True):
    
    model = model.to(device)
    scaler = GradScaler() if use_amp else None

    # Allowing PyTorch to choose fastest convolution algorithms to help with training speed
    torch.backends.cudnn.benchmark = True  
    
    for epoch in range(num_epochs):
        if hasattr(train_loader.dataset, "set_epoch_info"):
            train_loader.dataset.set_epoch_info(epoch, num_epochs)
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        start_time = time.time()

        for i, (inputs, labels) in enumerate(train_loader):
            print(f"Batch {i+1}/{len(train_loader)}")
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            if use_amp:
                with autocast(device_type=device.type):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        avg_train_loss = train_loss / total
        train_acc = correct / total

        # Validation phase
        model.eval()
        all_preds = []
        all_labels = []
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                with autocast(device_type=device.type) if use_amp else torch.no_grad():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_f1 = f1_score(all_labels, all_preds, average='weighted')

        print(f"Epoch {epoch+1:02d}: "
            f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, Val F1: {val_f1:.4f}, "
            f"Time: {time.time() - start_time:.1f}s")

        # Early stopping
        if early_stopping:
            early_stopping.step(val_f1, model)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                early_stopping.restore_best_weights(model)
                break

    return model, val_f1

# Trying out a different model structure to improve performance (Second model attempt)
class CNNBiLSTMClassifier2(nn.Module):
    def __init__(self, input_size=256, num_classes=4, dropout_p=0.4):
        super(CNNBiLSTMClassifier2, self).__init__()

        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, None))  
        )

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=128,
            num_layers=3,
            bidirectional=True,
            dropout=dropout_p,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)        
        x = x.squeeze(2)       
        x = x.permute(0, 2, 1) 

        lstm_out, _ = self.lstm(x)  
        x = lstm_out[:, -1, :]     

        return self.classifier(x)
    
# Third iteration of the model with an attention mechanism added
class CNNBiLSTMAttentionClassifier(nn.Module):
    def __init__(self, input_size=256, num_classes=4, dropout_p=0.4):
        super(CNNBiLSTMAttentionClassifier, self).__init__()

        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, None))  
        )

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=128,
            num_layers=3,
            bidirectional=True,
            dropout=dropout_p,
            batch_first=True
        )

        # Attention layer
        self.attention = nn.Linear(128 * 2, 1) 

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)        
        x = x.squeeze(2)       
        x = x.permute(0, 2, 1) 

        lstm_out, _ = self.lstm(x) 

        # Attention mechanism
        attn_weights = self.attention(lstm_out)             
        attn_weights = torch.softmax(attn_weights, dim=1)   
        context = torch.sum(attn_weights * lstm_out, dim=1) 

        return self.classifier(context)
    
# Fourth Iteration
class CNNBiLSTMAttentionClassifier2(nn.Module):
    def __init__(self, input_size=256, num_classes=4, dropout_p=0.4, lstm_hidden=256):
        super().__init__()

        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, None))  
        )

        self.dropout_cnn = nn.Dropout(dropout_p)

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden,
            num_layers=2,
            bidirectional=True,
            dropout=dropout_p,
            batch_first=True
        )

        # Attention layer (tanh for sharper focus)
        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_hidden * 2),
            nn.Linear(lstm_hidden * 2, 128),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.dropout_cnn(x)
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)

        lstm_out, _ = self.lstm(x)

        attn_weights = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)

        return self.classifier(context)
    
# Creating confusion matrix (both visual and printed) and classification report to visualize model's effectiveness at predicting each composer
def evaluate_confusion_matrix(model, dataloader, device, class_names, 
                               show_plot=True, show_text=True, show_report=True):
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)

    if show_text:
        cm_percent = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True) * 100

        print("\nConfusion Matrix (% per actual class):")
        header = f"{'':<12}" + "".join(f"{name:<12}" for name in class_names)
        print(header)
        print("-" * len(header))
        
        for i, row in enumerate(cm_percent):
            row_text = f"{class_names[i]:<12}" + "".join(f"{val:>10.1f}%" for val in row)
            print(row_text)

        print("\nNote: Rows = Actual, Columns = Predicted")

    if show_report:
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

        acc = accuracy_score(all_labels, all_preds)
        print(f"Validation Accuracy: {acc:.4f}")

    if show_plot:
        fig, ax = plt.subplots(figsize=(6, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=ax, cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.show()