"""
utils_ppatel.py
Payal Patel: feature extraction, model, eval, etc.
"""

import os
import shutil
import zipfile
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mido
import pretty_midi
import kagglehub
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

# Set device
gpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Download and extract dataset
def download_extract_dataset():
    path = kagglehub.dataset_download("blanderbuss/midi-classic-music")
    zip_path = os.path.join(path, 'midiclassics.zip')
    extract_path = os.path.join('data', 'kaggle', 'midiclassics')

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Extracted files to:", extract_path)
    return extract_path
"""
Clean Dataset
"""
# Use a portion of Antonio's code as a baseline to set ourselves up the same
# Clean dataset
TARGET_COMPOSERS = ['Bach', 'Beethoven', 'Chopin', 'Mozart']

# Keep directories that contain a target composer's name
def clean_dataset(extract_path):
    for item in os.listdir(extract_path):
        item_path = os.path.join(extract_path, item)
        if not any(composer.lower() in item.lower() for composer in TARGET_COMPOSERS):
            if os.path.isfile(item_path):
                os.remove(item_path)
                print(f"Deleted file: {item_path}")
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
                print(f"Deleted directory: {item_path}")

    # also delete "C.P.E.Bach" files. This was the son of J.S. Bach, and we want to keep only the main composers
    for item in os.listdir(extract_path):
        if 'C.P.E.Bach' in item:
            item_path = os.path.join(extract_path, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
                print(f"Deleted file: {item_path}")
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
                print(f"Deleted directory: {item_path}")

"""
Mido Feature Extraction
"""
def extract_features(midi_path, max_e=1000):
    try:
        # Load MIDI file
        midi_file = mido.MidiFile(midi_path)
        notes = []
        # Loop through MIDI files
        for e in midi_file:
            # Only the note being played
            if e.type == 'note_on' and e.velocity > 0:
                # Pitch (note), time, velocity
                notes.append((e.note, e.velocity, e.time))
            # Stop collecting after max events
            if len(notes) >= max_e:
                break
        if not notes:
            return None
        # Convert list to numpy array
        matrix_n = np.array(notes)
        # Pad with zeros if its less than the max. If more, cut.
        if matrix_n.shape[0] < max_e:
            padding = max_e - matrix_n.shape[0]
            matrix_n = np.pad(matrix_n, ((0, padding), (0, 0)), mode='constant')
        else:
            matrix_n = matrix_n[:max_e]
        # Flatten - 1D feature vector
        return matrix_n.flatten()
    except Exception as error:
        print(f"Skip {midi_path}: {error}")
        return None

"""
PrettyMIDI Feature Extraction
"""
# Feature extraction with pretty_midi for chords and tempo
def extract_chords_tempo(midi_path):
    try:
        # Load MIDI file using PrettyMIDI
        midi_obj = pretty_midi.PrettyMIDI(midi_path)
        # Extract tempo changes & calculate avg tempo (BPM)
        tempo = midi_obj.get_tempo_changes()[1]
        avg_tempo = np.mean(tempo) if len(tempo) > 0 else 0.0
        # Extract chroma feature, and the avg across all time frames
        chroma = midi_obj.get_chroma()
        chord = np.mean(chroma, axis=1)
        return chord, avg_tempo
    except Exception as error:
        print(f"Tempo and/or Chord extraction has failed {midi_path}: {error}")
        return None, None
    
"""
Tempo Distribution
"""
def plot_tempo_distribution(tempo_f, composer_label):
    tempos = np.array(tempo_f).flatten()
    labels = np.array(composer_label)
    
    plt.figure(figsize=(10,6))
    for composer in np.unique(labels):
        composer_tempo = tempos[labels == composer]
        plt.hist(composer_tempo, bins=30, alpha=0.5, label=composer)
    plt.title("Tempo Distribution")
    plt.xlabel("Tempo")
    plt.ylabel("Count")
    plt.legend()
    plt.show()

"""
Pitch mean distribution
"""
def plot_pitch_distribution(note_f, composer_label):
    note_features_reshaped = np.array(note_f).reshape(len(note_f), -1, 3)
    pitch_mean = np.mean(note_features_reshaped[:, :, 0], axis=1)
    lab = np.array(composer_label)

    plt.figure(figsize=(10, 6))
    for composer in np.unique(lab):
        comp_pitch = pitch_mean[lab == composer]
        plt.hist(comp_pitch, bins=30, alpha=0.5, label=composer)
    plt.title("Pitch Mean Distribution by Composer")
    plt.xlabel("Mean Pitch")
    plt.ylabel("Count")
    plt.legend()
    plt.show()

"""
Velocity mean distribution
"""
def plot_velocity_distribution(note_f, composer_label):
    note_features_reshaped = np.array(note_f).reshape(len(note_f), -1, 3)
    velocity_mean = np.mean(note_features_reshaped[:, :, 1], axis=1)
    labels = np.array(composer_label)
    
    plt.figure(figsize=(10, 6))
    for composer in np.unique(labels):
        comp_velocity = velocity_mean[labels == composer]
        plt.hist(comp_velocity, bins=30, alpha=0.5, label=composer)
    plt.title("Velocity Mean Distribution")
    plt.xlabel("Mean Velocity")
    plt.ylabel("Count")
    plt.legend()
    plt.show()

"""
Time Distribution
"""
def plot_time_distribution(note_f, composer_label):
    note_features_reshaped = np.array(note_f).reshape(len(note_f), -1, 3)
    time_mean = np.mean(note_features_reshaped[:, :, 2], axis=1)
    labels = np.array(composer_label)
    
    plt.figure(figsize=(10, 6))
    for composer in np.unique(labels):
        comp_time = time_mean[labels == composer]
        plt.hist(comp_time, bins=30, alpha=0.5, label=composer)
    plt.title("Time Mean Distribution")
    plt.xlabel("Mean Time")
    plt.ylabel("Count")
    plt.legend()
    plt.show()

"""
Data prep
"""
def extract_all_features(extract_path):
    # Initialize empty lists to store each feature
    note_f = [] # From Mido: pitch, velocity, time
    chord_f = [] # From PrettyMIDI: avg chroma
    tempo_f = [] # From PrettyMIDI: avg BPM
    composer_label = [] # Composer names

    # Loop through each composer
    for name in TARGET_COMPOSERS:
        composers = os.path.join(extract_path, name)
        
        # Loop through each file
        for feat_name in os.listdir(composers):
            if feat_name.endswith(".mid") or feat_name.endswith(".midi"):
                f_path = os.path.join(composers, feat_name)

                # Extract note features using mido
                feat_note = extract_features(f_path)
                # Extract tempo and chord features using prettymidi
                feat_chord, feat_tempo = extract_chords_tempo(f_path)

                # Append only if all feature extractions were successful
                if feat_note is not None and feat_chord is not None and feat_tempo is not None:
                    note_f.append(feat_note)
                    chord_f.append(feat_chord)
                    tempo_f.append([feat_tempo])
                    composer_label.append(name)
    # Single feature matrix
    X_full = np.hstack([np.array(note_f), np.array(chord_f), np.array(tempo_f)])
    
    # Convert the composer name to numerical labels using LabelEncoder
    le = LabelEncoder()
    y_full = le.fit_transform(composer_label)
    return X_full, y_full, le, note_f, chord_f, tempo_f, composer_label

"""
Train/test + more prep
"""
# Prepare train/test split and create dataloaders
def prepare_data(X_full, y_full, batch_size=32):
    # Initalize standard scaler for normalization
    scaler = StandardScaler()

    # normalize
    feat_scaled = scaler.fit_transform(X_full)

    # Split into train test (80/20)
    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        feat_scaled, y_full, test_size=0.2, random_state=42, stratify=y_full
    )

    # Convert to PyTorch tensors
    X_train_ten = torch.tensor(X_train_np, dtype=torch.float32)
    y_train_ten = torch.tensor(y_train_np, dtype=torch.long)
    X_test_ten = torch.tensor(X_test_np, dtype=torch.float32)
    y_test_ten = torch.tensor(y_test_np, dtype=torch.long)

    # Create pytorch tensordatasets for train/test
    train_data = TensorDataset(X_train_ten, y_train_ten)
    test_data = TensorDataset(X_test_ten, y_test_ten)

    # Dataloaders for processing
    train_load = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_load = DataLoader(test_data, batch_size=batch_size)

    return train_load, test_load, X_train_np, y_train_np, X_test_np, y_test_np

"""
Class weights
"""
# Compute class weights for imbalanced classes
def compute_class_weights(y_train_np):
    class_w = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_np),
        y=y_train_np
    )
    # Convert to tensor and move to GPU
    class_weights = torch.tensor(class_w, dtype=torch.float32).to(gpu_device)
    return class_weights

"""
Initialize, criterion, optimizer
"""
def get_model(input_feature, num_classes, labels, device, lr=0.001):
    model = ComposerCNNBiLSTMWithAttention(input_feature, num_classes)
    model.to(device)
    # Compute class weights (add in again bc idk why this file is being weird if i dont)
    class_weights_np = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32).to(device)
    # CrossEntropyLoss with class weights for imbalance
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    return model, criterion, optimizer

"""
CNN-BiLSTM Model with Attention
"""
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()

        # Compute attention scores - linear
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, encoder_outputs):
        # Compute attention score for each time
        score = self.attention(encoder_outputs)
        # Normalize scores to weights - softmax
        attn_weights = torch.softmax(score, dim=1)
        # Sum of encoder output using the attention weights
        context_vector = torch.sum(attn_weights * encoder_outputs, dim=1)
        return context_vector, attn_weights

class ComposerCNNBiLSTMWithAttention(nn.Module):
    def __init__(self, input_size, num_classes, dropout_prob=0.1):
        super(ComposerCNNBiLSTMWithAttention, self).__init__()
        # CNN Layers
        self.cnn_layer = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_prob),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_prob),

            nn.Conv1d(64, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_prob)
        )
        # Bidirectional LSTM
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, batch_first=True, bidirectional=True)
        
        # Attention method
        self.attention = Attention(hidden_dim=256*2)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(256*2, num_classes)
    
    # Rearrange, CNN output, dimension
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn_layer(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)

        # Apply attention
        attn_out, attn_weights = self.attention(lstm_out)
        out = self.dropout(attn_out)
        out = self.fc(out)
        return out

"""
Train and evaluate model
"""

def train_and_evaluate(model, train_load, test_load, criterion, optimizer, device, le, num_epochs=120):
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_feat, batch_label in train_load:
            batch_feat = batch_feat.to(device)
            batch_label = batch_label.to(device)

            optimizer.zero_grad()
            outputs = model(batch_feat)
            loss = criterion(outputs, batch_label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_load)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # Evaluate model
    model.eval()
    pred = []
    true = []

    with torch.no_grad():
        for batch_feat, batch_label in test_load:
            batch_feat = batch_feat.to(device)
            batch_label = batch_label.to(device)

            outputs = model(batch_feat)
            _, preds = torch.max(outputs, 1)

            pred.extend(preds.cpu().numpy())
            true.extend(batch_label.cpu().numpy())

    # Print metrics
    print(classification_report(true, pred, target_names=le.classes_))
    print(f"Model Accuracy: {accuracy_score(true, pred):.4f}")

    # Confusion matrix
    cm = confusion_matrix(true, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()