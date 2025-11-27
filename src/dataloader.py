from pathlib import Path
from typing import List

import symusic
from torch.utils.data import Dataset
import torch
from src.utils import CONTEXT_SIZE
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    mu_dt = 0.1226
    std_dt = (0.0568 ** 0.5)  # ≈ 0.2656

    mu_dur = 0.6158
    std_dur = (0.8221 ** 0.5)  # ≈ 0.8703

    mu_vel = 64.6879
    std_vel = (314.4664 ** 0.5)  # ≈ 18.57

    # BOS token: start_time=0, duration=0, pitch=0, velocity=0
    positions.append([0.0, 0.0, 0, 0])
    curr_time = 0
    for note in notes:
        start_time = float(note.start)
        delta_time = min(start_time - curr_time, 8)
        curr_time = float(note.start)
        end_time = float(note.end)
        duration = min(end_time - start_time, 8)
        pitch = int(note.pitch)
        velocity = int(note.velocity)

        positions.append([(delta_time-mu_dt)/std_dt, (duration-mu_dur)/std_dur, pitch, (velocity-mu_vel)/std_vel])


    return torch.tensor(positions, dtype=torch.float32)


from pathlib import Path
from typing import List

def _process_single_file(file_path: Path):
    """
    Worker function that processes a single MIDI file and returns
    a position tensor or None if it fails or should be skipped.
    """
    try:
        score = symusic.Score.from_file(str(file_path))

        merge_score_tracks(score)
        score = score.to("second")

        if not score.tracks or len(score.tracks[0].notes) == 0:
            return None

        track = score.tracks[0]
        all_notes = list(track.notes)

        if len(all_notes) < 5:
            return None

        all_notes.sort(key=lambda x: x.start)

        position_tensors = _create_position_tensors(all_notes)
        return position_tensors.numpy()

    except Exception as e:
        print(f"Failed to process {file_path}: {e}")
        return None


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

    def _load_and_concatenate_files(self, file_list: List[Path], batch_size=50000):
        """Load all MIDI files in parallel in batches and concatenate."""

        all_tensors = []

        # Break file_list into batches
        for start in range(0, len(file_list), batch_size):
            batch_files = file_list[start : start + batch_size]

            print(f"Processing batch {start//batch_size+1} / { (len(file_list)-1)//batch_size + 1 }")

            with ProcessPoolExecutor(max_workers=20) as executor:
                futures = {executor.submit(_process_single_file, fp): fp for fp in batch_files}

                for i, future in enumerate(as_completed(futures)):
                    fp = futures[future]
                    print(f"  file {i+1}/{len(batch_files)}")

                    result = future.result()

                    if result is not None:
                        # result is numpy, convert back to torch here
                        all_tensors.append(torch.from_numpy(result))

            # Force cleanup between batches
            import gc
            gc.collect()
            torch.cuda.empty_cache()

        # Final concatenation
        if all_tensors:
            return torch.cat(all_tensors, dim=0)
        else:
            return torch.zeros((0, 4), dtype=torch.float32)

    def _create_chunks(self, all_position_tensors):
        """Split concatenated sequence into fixed-size chunks."""
        total_len = all_position_tensors.shape[0]
        PAD = torch.zeros(4, dtype=torch.float32)

        for i in range(0, total_len, self.max_seq_len):
            print(total_len // self.max_seq_len)
            print(i // self.max_seq_len)
            end_idx = min(i + self.max_seq_len, total_len)
            chunk = all_position_tensors[i:end_idx]
            original_len = chunk.shape[0]

            if original_len < self.max_seq_len:
                pad_amount = self.max_seq_len - original_len
                pad_tokens = PAD.unsqueeze(0).repeat(pad_amount, 1)  # (pad_amount, 131)
                chunk = torch.cat([chunk, pad_tokens], dim=0)

            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = torch.zeros(self.max_seq_len, dtype=torch.long)
            attention_mask[:original_len] = 1

            labels = chunk[1:].clone()  # Next position prediction
            last_position = chunk[-1:].clone()
            labels = torch.cat([labels, last_position], dim=0)
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