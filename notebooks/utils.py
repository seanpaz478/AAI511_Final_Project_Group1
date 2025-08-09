"""
Musical Composer Classification Utilities

This module contains reusable utility functions for processing MIDI files 
and working with musical data for composer classification.

Key Components:
- Data Processing: MIDI to piano roll conversion, segmentation, normalization
- Feature Extraction: Comprehensive musical feature computation
- Data Loading: Dataset loading and balancing utilities
- Dataset Classes: PyTorch dataset implementations
- General Utilities: Device detection, data loaders, visualization

Note: Model architectures and training functions are kept in separate 
model-specific notebooks for better organization.

Usage:
    from utils import *
    
    # Load and process data
    data, labels, features = load_segmented_dataset_no_overlap(...)
    
    # Create multimodal dataset
    dataset = MultimodalDataset(data, features, labels)
"""

import os
import shutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pretty_midi
from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import zipfile
import pickle
import numpy as np


# needed because some systems may not have np.int defined
if not hasattr(np, 'int'):
    np.int = int

# =============================================================================
# Dataset classes
# =============================================================================
class MultimodalDataset(Dataset):
    """
    Dataset class that handles both piano rolls (for CNN) and musical features (for MLP)
    """
    def __init__(self, piano_rolls, features_list, labels):
        # Convert piano rolls to tensor
        self.piano_rolls = torch.tensor(piano_rolls, dtype=torch.float32)

        # Convert feature dictionaries to feature vectors
        self.features = self._process_features(features_list)

        # Convert labels to tensor
        self.labels = torch.tensor(labels, dtype=torch.long)

        print(f"Multimodal Dataset Created:")
        print(f"Piano rolls: {self.piano_rolls.shape}")
        print(f"Features: {self.features.shape}")
        print(f"Labels: {self.labels.shape}")
        print(f"Total samples: {len(self.labels)}")


    def _process_features(self, features_list):
        """Convert list of feature dictionaries to tensor"""
        # Get feature names from first sample
        feature_names = list(features_list[0].keys())

        # Extract feature values for all samples
        feature_matrix = []
        for feature_dict in features_list:
            feature_vector = [feature_dict[name] for name in feature_names]
            feature_matrix.append(feature_vector)

        # Convert to tensor and normalize
        features_tensor = torch.tensor(feature_matrix, dtype=torch.float32)

        # Normalize features (important for MLP training)
        features_mean = features_tensor.mean(dim=0)
        features_std = features_tensor.std(dim=0)
        features_std[features_std == 0] = 1  # Avoid division by zero
        features_normalized = (features_tensor - features_mean) / features_std

        print(f"Feature Processing:")
        print(f"• Feature names: {feature_names[:5]}...")
        print(f"• Features normalized: mean≈0, std≈1")

        return features_normalized

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Piano roll with channel dimension for CNN: (1, 128, T)
        piano_roll = self.piano_rolls[idx].unsqueeze(0)

        # Feature vector for MLP: (num_features,)
        features = self.features[idx]

        # Label
        label = self.labels[idx]

        return piano_roll, features, label


# =============================================================================
# Data processing
# =============================================================================

def download_dataset():
    TARGET_COMPOSERS = ['Bach', 'Beethoven', 'Chopin', 'Mozart']
    path = kagglehub.dataset_download("blanderbuss/midi-classic-music")
    zip_path = os.path.join(path, 'midiclassics.zip')
    extract_path = os.path.join('..', 'data', 'kaggle', 'midiclassics')

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    # list files in extract_path that contain the target composers in name
    for composer in TARGET_COMPOSERS:
        composer_files = [f for f in os.listdir(extract_path) if composer.lower() in f.lower()]

    # Only keep directories that contain a target composer's name
    for item in os.listdir(extract_path):
        item_path = os.path.join(extract_path, item)
        if not any(composer.lower() in item.lower() for composer in TARGET_COMPOSERS):
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)

    # also delete "C.P.E.Bach" files. This was the son of J.S. Bach, and we want to keep only the main composers
    for item in os.listdir(extract_path):
        if 'C.P.E.Bach' in item:
            item_path = os.path.join(extract_path, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
    print(f"Dataset downloaded and extracted to {extract_path}")
    return {
        "path": path,
        "zip_path": zip_path,
        "extract_path": extract_path,
    }


def normalize_piano_roll(piano_roll):
    """
    Apply musical normalization to piano roll
    
    Args:
        piano_roll (np.ndarray): Piano roll array (128, T)
        
    Returns:
        np.ndarray: Normalized piano roll
    """
    # 1. Velocity normalization (already 0-1 from pretty_midi)
    normalized = piano_roll.copy()

    # 2. Optional: Focus on active pitch range
    active_pitches = np.any(normalized > 0, axis=1)
    if np.any(active_pitches):
        first_active = np.argmax(active_pitches)
        last_active = len(active_pitches) - 1 - np.argmax(active_pitches[::-1])

        # Ensure we keep a reasonable range (at least 60 semitones = 5 octaves)
        min_range = 60
        current_range = last_active - first_active + 1

        if current_range < min_range:
            expand = (min_range - current_range) // 2
            first_active = max(0, first_active - expand)
            last_active = min(127, last_active + expand)

    return normalized


def get_piano_roll_segments_no_overlap(midi_path, fs=100, segment_duration=20.0):
    """
    Extract NON-OVERLAPPING segments from a single MIDI file
    This preserves temporal relationships without data leakage
    
    Args:
        midi_path (str): Path to MIDI file
        fs (int): Sampling frequency
        segment_duration (float): Duration of each segment in seconds
        
    Returns:
        list: List of segment dictionaries with piano_roll, metadata, or None if failed
    """
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
        actual_duration = pm.get_end_time()

        # Skip very short pieces
        if actual_duration < segment_duration:
            return None

        segments = []
        segment_length = int(segment_duration * fs)

        # Extract non-overlapping segments
        current_time = 0.0
        segment_idx = 0

        while current_time + segment_duration <= actual_duration:
            end_time = current_time + segment_duration

            # Create segment
            pm_segment = pretty_midi.PrettyMIDI()
            for instrument in pm.instruments:
                new_instrument = pretty_midi.Instrument(
                    program=instrument.program,
                    is_drum=instrument.is_drum,
                    name=instrument.name
                )

                for note in instrument.notes:
                    if current_time <= note.start < end_time:
                        new_note = pretty_midi.Note(
                            velocity=note.velocity,
                            pitch=note.pitch,
                            start=note.start - current_time,
                            end=min(note.end - current_time, segment_duration)
                        )
                        new_instrument.notes.append(new_note)

                if new_instrument.notes:
                    pm_segment.instruments.append(new_instrument)

            # Convert to piano roll
            if pm_segment.instruments:
                piano_roll = pm_segment.get_piano_roll(fs=fs)

                # Ensure exact length
                if piano_roll.shape[1] > segment_length:
                    piano_roll = piano_roll[:, :segment_length]
                elif piano_roll.shape[1] < segment_length:
                    pad_width = segment_length - piano_roll.shape[1]
                    piano_roll = np.pad(piano_roll, ((0,0),(0,pad_width)), mode='constant')

                # Store segment with metadata
                segment_info = {
                    'piano_roll': piano_roll,
                    'song_id': midi_path,
                    'segment_idx': segment_idx,
                    'start_time': current_time
                }
                segments.append(segment_info)
                segment_idx += 1

            # Move to next segment (NO OVERLAP)
            current_time += segment_duration

        return segments if segments else None

    except Exception as e:
        print(f"Error processing {midi_path}: {e}")
        return None


# =============================================================================
# Feature extraction utils
# =============================================================================

def extract_musical_features(piano_roll):
    """
    Extract basic features that capture musical style
    
    Args:
        piano_roll (np.ndarray): Piano roll array (128, T)
        
    Returns:
        dict: Dictionary of musical features
    """
    features = {}

    # Temporal features
    note_density_timeline = np.sum(piano_roll > 0, axis=0)
    features['avg_notes_per_time'] = np.mean(note_density_timeline)
    features['note_density_variance'] = np.var(note_density_timeline)

    # Pitch features
    pitch_activity = np.sum(piano_roll > 0, axis=1)
    active_pitches = pitch_activity > 0
    if np.any(active_pitches):
        features['pitch_range'] = np.sum(active_pitches)
        features['lowest_pitch'] = np.argmax(active_pitches)
        features['highest_pitch'] = 127 - np.argmax(active_pitches[::-1])
    else:
        features['pitch_range'] = 0
        features['lowest_pitch'] = 60  # Middle C
        features['highest_pitch'] = 60

    # Rhythmic features
    onset_pattern = np.diff(note_density_timeline > 0).astype(int)
    features['onset_density'] = np.sum(onset_pattern == 1) / len(onset_pattern)

    return features


def extract_comprehensive_musical_features(piano_roll):
    """
    Extract comprehensive musical features for the MLP stream
    
    Args:
        piano_roll (np.ndarray): Piano roll array (128, T)
        
    Returns:
        dict: Dictionary of comprehensive musical features
    """


    features = {}

    # Get basic timeline and pitch data
    note_density_timeline = np.sum(piano_roll > 0, axis=0)
    pitch_activity = np.sum(piano_roll > 0, axis=1)
    active_pitches = pitch_activity > 0

    # ==========================================
    # Temporal/rhythmic features
    # ==========================================

    # Basic temporal statistics
    features['avg_notes_per_time'] = np.mean(note_density_timeline)
    features['note_density_variance'] = np.var(note_density_timeline)
    features['note_density_std'] = np.std(note_density_timeline)
    features['max_simultaneous_notes'] = np.max(note_density_timeline)

    # Rhythmic complexity
    onset_pattern = np.diff(note_density_timeline > 0).astype(int)
    features['onset_density'] = np.sum(onset_pattern == 1) / len(onset_pattern) if len(onset_pattern) > 0 else 0
    features['silence_ratio'] = np.sum(note_density_timeline == 0) / len(note_density_timeline)

    # Temporal distribution analysis
    active_frames = note_density_timeline > 0
    if np.any(active_frames):
        features['temporal_sparsity'] = 1 - (np.sum(active_frames) / len(active_frames))

        # Find note clusters (bursts of activity)
        cluster_changes = np.diff(active_frames.astype(int))
        features['activity_bursts'] = np.sum(cluster_changes == 1) / len(cluster_changes) if len(cluster_changes) > 0 else 0
    else:
        features['temporal_sparsity'] = 1.0
        features['activity_bursts'] = 0.0

    # ==========================================
    # Pitch/harmonic features
    # ==========================================

    if np.any(active_pitches):
        # Basic pitch statistics
        features['pitch_range'] = np.sum(active_pitches)
        features['lowest_pitch'] = np.argmax(active_pitches)
        features['highest_pitch'] = 127 - np.argmax(active_pitches[::-1])
        features['pitch_span'] = features['highest_pitch'] - features['lowest_pitch']

        # Pitch distribution
        weighted_pitches = np.arange(128) * pitch_activity
        total_weight = np.sum(pitch_activity)
        if total_weight > 0:
            features['pitch_centroid'] = np.sum(weighted_pitches) / total_weight
            features['pitch_variance'] = np.var(pitch_activity[active_pitches])
        else:
            features['pitch_centroid'] = 60  # Middle C
            features['pitch_variance'] = 0

        # Register analysis (musical ranges)
        bass_range = pitch_activity[21:48]  # A0 to B2
        mid_range = pitch_activity[48:72]   # C3 to B4
        treble_range = pitch_activity[72:108] # C5 to B7

        total_activity = np.sum(pitch_activity)
        features['bass_activity'] = np.sum(bass_range) / total_activity if total_activity > 0 else 0
        features['mid_activity'] = np.sum(mid_range) / total_activity if total_activity > 0 else 0
        features['treble_activity'] = np.sum(treble_range) / total_activity if total_activity > 0 else 0

    else:
        # Default values for empty piano rolls
        features.update({
            'pitch_range': 0, 'lowest_pitch': 60, 'highest_pitch': 60,
            'pitch_span': 0, 'pitch_centroid': 60, 'pitch_variance': 0,
            'bass_activity': 0, 'mid_activity': 0, 'treble_activity': 0
        })

    # ==========================================
    # Other harmonic features
    # ==========================================

    # Analyze simultaneous note patterns (chords vs single notes)
    chord_frames = note_density_timeline >= 3  # 3+ simultaneous notes = chord
    single_note_frames = note_density_timeline == 1

    features['chord_ratio'] = np.sum(chord_frames) / len(note_density_timeline)
    features['single_note_ratio'] = np.sum(single_note_frames) / len(note_density_timeline)
    features['polyphony_complexity'] = np.mean(note_density_timeline[note_density_timeline > 0]) if np.any(note_density_timeline > 0) else 0

    # Chord complexity analysis
    if np.sum(chord_frames) > 0:
        chord_complexities = note_density_timeline[chord_frames]
        features['avg_chord_size'] = np.mean(chord_complexities)
        features['chord_variance'] = np.var(chord_complexities)
    else:
        features['avg_chord_size'] = 0
        features['chord_variance'] = 0

    # ==========================================
    # Velocity/dynamics features
    # ==========================================

    if np.any(piano_roll > 0):
        velocities = piano_roll[piano_roll > 0]
        features['avg_velocity'] = np.mean(velocities)
        features['velocity_variance'] = np.var(velocities)
        features['velocity_range'] = np.max(velocities) - np.min(velocities)
        features['dynamic_complexity'] = len(np.unique(velocities)) / len(velocities)
    else:
        features.update({
            'avg_velocity': 0, 'velocity_variance': 0,
            'velocity_range': 0, 'dynamic_complexity': 0
        })

    # ==========================================
    # Style-specific features
    # ==========================================

    # Measure musical "busyness" vs "spaciousness"
    features['overall_density'] = np.sum(piano_roll > 0) / piano_roll.size

    # Temporal consistency (how regular/irregular the rhythm is)
    if len(note_density_timeline) > 1:
        features['rhythmic_regularity'] = 1 / (1 + np.var(note_density_timeline))
    else:
        features['rhythmic_regularity'] = 1.0

    # Pitch movement patterns
    if np.sum(active_pitches) > 1:
        pitch_centers = []
        for t in range(piano_roll.shape[1]):
            frame = piano_roll[:, t]
            if np.any(frame > 0):
                weighted_pitch = np.sum(np.arange(128) * frame) / np.sum(frame)
                pitch_centers.append(weighted_pitch)

        if len(pitch_centers) > 1:
            pitch_movement = np.diff(pitch_centers)
            features['pitch_movement_variance'] = np.var(pitch_movement)
            features['melodic_direction_changes'] = np.sum(np.diff(np.sign(pitch_movement)) != 0) / len(pitch_movement) if len(pitch_movement) > 0 else 0
        else:
            features['pitch_movement_variance'] = 0
            features['melodic_direction_changes'] = 0
    else:
        features['pitch_movement_variance'] = 0
        features['melodic_direction_changes'] = 0

    return features

def load_comprehensive_features(data, cache_path="comprehensive_musical_features.pkl"):
    """
    Load comprehensive features from cache if available, else compute and cache them.
    Args:
        data (list or np.ndarray): List/array of piano rolls
        cache_path (str): Path to cache file (will be saved in ../local_cache/)
    Returns:
        list: List of comprehensive feature dicts
    """
    cache_dir = "../local_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, os.path.basename(cache_path))
    if os.path.exists(cache_path):
        print(f"Loading comprehensive_features from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            comprehensive_features = pickle.load(f)
    else:
        comprehensive_features = [extract_comprehensive_musical_features(p) for p in data]
        with open(cache_path, "wb") as f:
            pickle.dump(comprehensive_features, f)
        print(f"Saved comprehensive_features to cache: {cache_path}")
    return comprehensive_features


# =============================================================================
# Data/processing loading utils
# =============================================================================

def balance_classes_song_aware(segments_by_composer, target_samples_per_class=None):
    """
    Balance classes by selecting complete songs (with all their segments)
    This preserves temporal relationships within songs
    
    Args:
        segments_by_composer (dict): Dictionary mapping composer to list of segments
        target_samples_per_class (int, optional): Target number of samples per class
        
    Returns:
        dict: Balanced segments by composer
    """
    # Calculate current distribution
    composer_counts = {composer: len(segments) for composer, segments in segments_by_composer.items()}
    max_count = max(composer_counts.values()) if target_samples_per_class is None else target_samples_per_class

    print(f"Original segment distribution: {composer_counts}")
    print(f"Target segments per class: {max_count}")

    balanced_segments = {}

    for composer, segments in segments_by_composer.items():
        current_count = len(segments)

        if current_count >= max_count:
            # If we have enough, randomly select songs to reach target
            songs_dict = defaultdict(list)
            for segment in segments:
                songs_dict[segment['song_id']].append(segment)

            # Randomly select songs until we reach target count
            selected_segments = []
            song_ids = list(songs_dict.keys())
            np.random.shuffle(song_ids)

            for song_id in song_ids:
                song_segments = songs_dict[song_id]
                if len(selected_segments) + len(song_segments) <= max_count:
                    selected_segments.extend(song_segments)
                elif len(selected_segments) < max_count:
                    # Partial song - take contiguous segments from the beginning
                    needed = max_count - len(selected_segments)
                    selected_segments.extend(song_segments[:needed])
                    break

            balanced_segments[composer] = selected_segments
            print(f"  {composer}: {current_count} → {len(selected_segments)} (downsampled)")

        else:
            # Need to oversample - duplicate entire songs
            needed_samples = max_count - current_count

            # Group segments by song
            songs_dict = defaultdict(list)
            for segment in segments:
                songs_dict[segment['song_id']].append(segment)

            song_ids = list(songs_dict.keys())
            selected_segments = segments.copy()  # Start with all original segments

            # Add complete songs until we reach target
            while len(selected_segments) < max_count:
                # Randomly select a song to duplicate
                song_id = np.random.choice(song_ids)
                song_segments = songs_dict[song_id]

                if len(selected_segments) + len(song_segments) <= max_count:
                    # Add entire song
                    for segment in song_segments:
                        # Create a copy with new metadata to avoid conflicts
                        new_segment = segment.copy()
                        new_segment['song_id'] = f"{segment['song_id']}_dup_{len(selected_segments)}"
                        selected_segments.append(new_segment)
                else:
                    # Add partial song if needed
                    remaining = max_count - len(selected_segments)
                    for i, segment in enumerate(song_segments[:remaining]):
                        new_segment = segment.copy()
                        new_segment['song_id'] = f"{segment['song_id']}_dup_{len(selected_segments)}"
                        selected_segments.append(new_segment)
                    break

            balanced_segments[composer] = selected_segments
            print(f"{composer}: {current_count} → {len(selected_segments)} (+{len(selected_segments) - current_count} from song duplication)")

    return balanced_segments


def load_segmented_dataset_no_overlap(extract_path, target_composers, segment_duration=20.0,
                                     max_files_per_composer=None, balance_classes=True):
    """
    Load dataset with NON-OVERLAPPING segmentation and balancing
    
    Args:
        extract_path (str): Path to extracted MIDI files
        target_composers (list): List of composer names
        segment_duration (float): Duration of each segment in seconds
        max_files_per_composer (int, optional): Max files to process per composer
        balance_classes (bool): Whether to balance classes
        
    Returns:
        tuple: (data, labels, features) or (None, None, None) if failed
    """
    print("LOADING DATASET WITH NON-OVERLAPPING SEGMENTATION...")
    print(f"Segment duration: {segment_duration}s with NO OVERLAP")
    print(f"Balance classes: {balance_classes}")

    cache_dir = "../local_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"segmented_dataset_no_overlap_{segment_duration}.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            print(f"Loading segmented dataset from cache: {cache_path}")
            return pickle.load(f)

    composer_to_idx = {c: i for i, c in enumerate(target_composers)}
    segments_by_composer = {composer: [] for composer in target_composers}

    for composer in target_composers:
        print(f"\n--- Processing {composer} ---")
        composer_dir = os.path.join(extract_path, composer)

        if not os.path.isdir(composer_dir):
            print(f"Directory not found: {composer_dir}")
            continue

        files_processed = 0
        total_segments = 0

        midi_files = [f for f in os.listdir(composer_dir)
                     if f.lower().endswith(('.mid', '.midi'))]

        if max_files_per_composer:
            midi_files = midi_files[:max_files_per_composer]

        for file in midi_files:
            midi_path = os.path.join(composer_dir, file)

            try:
                # Extract NON-OVERLAPPING segments from this file
                segments = get_piano_roll_segments_no_overlap(
                    midi_path,
                    segment_duration=segment_duration
                )

                if segments is None:
                    continue

                # Process each segment
                valid_segments = []
                for segment_info in segments:
                    piano_roll = segment_info['piano_roll']

                    # Normalize the segment
                    normalized_segment = normalize_piano_roll(piano_roll)

                    # Extract musical features
                    features = extract_musical_features(normalized_segment)

                    # Quality check: skip if too sparse
                    note_density = features['avg_notes_per_time']
                    if note_density < 0.05:  # Very sparse, likely poor quality
                        continue

                    # Update segment info
                    segment_info['piano_roll'] = normalized_segment
                    segment_info['features'] = features
                    segment_info['label'] = composer_to_idx[composer]

                    valid_segments.append(segment_info)

                # Add valid segments to composer collection
                segments_by_composer[composer].extend(valid_segments)
                total_segments += len(valid_segments)
                files_processed += 1

                if files_processed % 10 == 0:
                    print(f"  Processed {files_processed} files, created {total_segments} segments...")

            except Exception as e:
                print(f"  Error processing {file}: {e}")
                continue

        print(f"{composer}: {files_processed} files → {total_segments} segments")

    # Balance classes if requested
    if balance_classes:
        print(f"\nBALANCING CLASSES (SONG-AWARE, NO OVERLAP)...")
        segments_by_composer = balance_classes_song_aware(segments_by_composer)

    # Convert to arrays
    all_data = []
    all_labels = []
    all_features = []
    all_song_ids = []

    for composer, segments in segments_by_composer.items():
        for segment_info in segments:
            all_data.append(segment_info['piano_roll'])
            all_labels.append(segment_info['label'])
            all_features.append(segment_info['features'])
            # Append unique song_id with segment index
            unique_song_id = f"{segment_info['song_id']}__{segment_info['segment_idx']}"
            all_song_ids.append(unique_song_id)

    data = np.array(all_data)
    labels = np.array(all_labels)

    print(f"Total samples: {len(data)}")
    print(f"Data shape: {data.shape}")
    final_counts = np.bincount(labels)
    for i, composer in enumerate(target_composers):
        if i < len(final_counts):
            print(f"  {composer}: {final_counts[i]} samples")

    # Save to cache
    with open(cache_path, 'wb') as f:
        pickle.dump((data, labels, all_features, all_song_ids), f)

    return data, labels, all_features, all_song_ids


# =============================================================================
# Other utils
# =============================================================================

def get_device():
    """
    Get the best available device for training
    
    Returns:
        torch.device: Best available device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 Using CUDA GPU: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("🚀 Using CPU")
    
    return device


def split_dataset_by_song(song_ids, train_split=0.8, seed=42):
    """
    Split dataset so that all segments from the same song are assigned to either train or validation.
    Args:
        segments (np.ndarray): Array of piano roll segments
        labels (np.ndarray): Array of labels
        features (list): List of feature dicts
        song_ids (list): List of song_id for each segment
        train_split (float): Fraction of data for training
        seed (int): Random seed for reproducibility
    Returns:
        (train_idx, val_idx): Indices for train and validation sets
    """
    np.random.seed(seed)
    # Get unique song ids
    unique_songs = np.unique(song_ids)
    np.random.shuffle(unique_songs)
    n_train = int(len(unique_songs) * train_split)
    train_songs = set(unique_songs[:n_train])
    val_songs = set(unique_songs[n_train:])
    train_idx = [i for i, sid in enumerate(song_ids) if sid in train_songs]
    val_idx = [i for i, sid in enumerate(song_ids) if sid in val_songs]
    return train_idx, val_idx


def create_songwise_data_loaders(data, labels, features, song_ids, batch_size=32, train_split=0.8, seed=42, dataset_class=MultimodalDataset):
    """
    Create train and validation data loaders with a song-level split.
    Args:
        data (np.ndarray): Array of piano roll segments
        labels (np.ndarray): Array of labels
        features (list): List of feature dicts
        song_ids (list): List of song_id for each segment
        batch_size (int): Batch size
        train_split (float): Fraction of data for training
        seed (int): Random seed
    Returns:
        (train_loader, val_loader)
    """
    train_idx, val_idx = split_dataset_by_song(song_ids, train_split, seed)
    # Subset data
    train_data = data[train_idx]
    train_labels = labels[train_idx]
    train_features = [features[i] for i in train_idx]
    val_data = data[val_idx]
    val_labels = labels[val_idx]
    val_features = [features[i] for i in val_idx]
    # Create datasets
    train_dataset = dataset_class(train_data, train_features, train_labels)
    val_dataset = dataset_class(val_data, val_features, val_labels)
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(f"Songwise Data Loaders Created:")
    print(f"• Training samples: {len(train_dataset)}")
    print(f"• Validation samples: {len(val_dataset)}")
    print(f"• Batch size: {batch_size}")
    return train_loader, val_loader


def plot_training_curves(train_losses, val_accuracies, title_prefix="Model"):
    """
    Plot training loss and validation accuracy curves
    
    Args:
        train_losses (list): Training losses per epoch
        val_accuracies (list): Validation accuracies per epoch
        title_prefix (str): Prefix for plot titles
    """
    plt.figure(figsize=(12, 4))
    
    # Training Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-', linewidth=2, label='Training Loss')
    plt.title(f'{title_prefix} - Training Loss', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Validation Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, 'g-', linewidth=2, label='Validation Accuracy')
    plt.title(f'{title_prefix} - Validation Accuracy', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def evaluate_model_comprehensive(model, val_loader, target_composers, device='cpu'):
    """
    Generate comprehensive evaluation metrics for any model
    
    Args:
        model: Trained PyTorch model
        val_loader: Validation data loader
        target_composers (list): List of composer names
        device (str): Device to evaluate on
        
    Returns:
        dict: Comprehensive evaluation results
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []

    print("EVALUATING MODEL...")

    with torch.no_grad():
        for batch_data in val_loader:
            # Handle different dataset types
            if len(batch_data) == 2:  # Simple dataset (data, labels)
                data, labels = batch_data
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
            elif len(batch_data) == 3:  # Multimodal dataset (piano_rolls, features, labels)
                piano_rolls, features, labels = batch_data
                piano_rolls, features, labels = piano_rolls.to(device), features.to(device), labels.to(device)
                outputs = model(piano_rolls, features)
            
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)

    # Classification Report
    print("\nCLASSIFICATION REPORT:")
    print("=" * 60)
    report = classification_report(
        all_labels,
        all_predictions,
        target_names=target_composers,
        digits=4
    )
    print(report)

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)

    # Plot Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=target_composers,
        yticklabels=target_composers,
        cbar_kws={'label': 'Number of Samples'}
    )
    plt.title('Model - Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Composer', fontsize=12)
    plt.ylabel('True Composer', fontsize=12)
    plt.tight_layout()
    plt.show()

    # Per-class accuracy
    print("\nPER-CLASS ACCURACY:")
    print("=" * 40)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    for i, composer in enumerate(target_composers):
        print(f"{composer:12}: {class_accuracies[i]:.4f} ({class_accuracies[i]*100:.2f}%)")

    # Overall metrics
    overall_accuracy = np.sum(cm.diagonal()) / np.sum(cm)
    print(f"\nOVERALL ACCURACY: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")

    # Most confused pairs
    print("\nMOST CONFUSED PAIRS:")
    print("=" * 40)
    confusion_pairs = []
    for i in range(len(target_composers)):
        for j in range(len(target_composers)):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append((target_composers[i], target_composers[j], cm[i, j]))

    confusion_pairs.sort(key=lambda x: x[2], reverse=True)
    for true_composer, pred_composer, count in confusion_pairs[:5]:
        print(f"{true_composer} → {pred_composer}: {count} samples")

    return {
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities,
        'confusion_matrix': cm,
        'classification_report': report,
        'overall_accuracy': overall_accuracy,
        'class_accuracies': class_accuracies
    }