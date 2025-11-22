from pathlib import Path
from typing import List

import symusic
from torch.utils.data import Dataset
import torch
from src.train_transformer import CONTEXT_SIZE


def merge_score_tracks(score: symusic.Score) -> None:
    """
    Merge tracks in a score by combining their notes into a single track.
    """
    notes = []
    for track in score.tracks:
        for note in track.notes:
            notes.append(note)
    score.tracks.clear()
    track = symusic.Track()
    score.tracks.append(track)
    for note in notes:
        track.notes.append(note)


def _create_position_tensors(notes) -> torch.Tensor:
    """Create 4D position tensors [start_time, duration, pitch, velocity]."""
    positions = []

    # BOS token: start_time=0, duration=0, pitch=0, velocity=0
    positions.append([0.0, 0.0, 0, 0])
    curr_time = 0
    for note in notes:
        start_time = float(note.start)
        delta_time = max(start_time - curr_time, 8)
        curr_time = float(note.start)
        end_time = float(note.end)
        duration = max(end_time - start_time, 8)
        pitch = int(note.pitch)
        velocity = int(note.velocity)

        positions.append([delta_time, duration, pitch, velocity])


    return torch.tensor(positions, dtype=torch.float32)


class MidiDataset4D(Dataset):
    """Dataset that concatenates all MIDI files and chunks for pretraining."""

    def __init__(self, files: List[Path], max_seq_len: int = CONTEXT_SIZE):
        self.files = files
        self.max_seq_len = max_seq_len
        self.chunks = []

        # Load all files and create one big concatenated sequence
        print("Loading and concatenating all MIDI files...")
        all_position_tensors = self._load_and_concatenate_files(self.files)

        # Chunk the concatenated sequence
        print(f"Chunking into sequences of length {max_seq_len}...")
        self._create_chunks(all_position_tensors)

        print(f"Created {len(self.chunks)} chunks for training")

    def _load_and_concatenate_files(self, file_list: List[Path]):
        """Load all MIDI files and concatenate into one big sequence."""
        all_tensors = []

        for file_path in file_list:
            try:
                # Load MIDI file using symusic
                score = symusic.Score.from_file(str(file_path))

                # Use preprocessing functions to clean up the score (in tick format)
                merge_score_tracks(score)

                # Convert to seconds after preprocessing
                score = score.to("second")

                # Extract notes from the merged track
                if not score.tracks or len(score.tracks[0].notes) == 0:
                    continue

                track = score.tracks[0]
                all_notes = list(track.notes)

                # Skip very short pieces
                if len(all_notes) < 5:
                    continue

                # Sort notes by start time
                all_notes.sort(key=lambda x: x.start)

                # Create 4D position tensors [start_time, duration, pitch, velocity]
                position_tensors = _create_position_tensors(all_notes)

                # Add this piece to the concatenated sequence
                all_tensors.append(position_tensors)

            except Exception as e:
                print(f"Failed to process {file_path}: {e}")
                continue

        # Concatenate all pieces
        if all_tensors:
            concatenated = torch.cat(all_tensors, dim=0)
            return concatenated
        else:
            return torch.tensor([], dtype=torch.float32).reshape(0, 4)

    def _create_chunks(self, all_position_tensors):
        """Split concatenated sequence into fixed-size chunks."""
        total_len = all_position_tensors.shape[0]

        for i in range(0, total_len, self.max_seq_len):
            print(total_len // self.max_seq_len)
            print(i // self.max_seq_len)
            end_idx = min(i + self.max_seq_len, total_len)
            chunk = all_position_tensors[i:end_idx]
            original_len = chunk.shape[0]

            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = torch.ones(self.max_seq_len, dtype=torch.long)

            # Pad chunk to max_seq_len if it's the last chunk and shorter
            if chunk.shape[0] < self.max_seq_len:
                pad_len = self.max_seq_len - chunk.shape[0]
                last_time = chunk[-1, 0].item()
                pad_tensor = torch.tensor([last_time, 0.0, 2, 0]).repeat(pad_len, 1)
                chunk = torch.cat([chunk, pad_tensor], dim=0)

                # Mask the padded positions
                attention_mask[original_len:] = 0

            labels = chunk[1:].clone()  # Next position prediction
            last_position = chunk[-1:].clone()
            labels = torch.cat([labels, last_position], dim=0)

            if original_len < self.max_seq_len:
                labels[original_len:] = -100

            self.chunks.append({
                'input_ids': chunk,
                'labels': labels,
                'attention_mask': attention_mask
            })

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]