from pathlib import Path

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset
import lightning as pl
import torch

from sklearn.model_selection import train_test_split

from src.modeling import MidiQwenNew
from src.dataloader import MidiDataset4D
from src.transformer import MidiAria
from src.utils import EPOCHS

BATCH_SIZE = 32
CONTEXT_SIZE = 4096
MAX_SEQ_LEN = CONTEXT_SIZE

def custom_collate_fn(batch):
    """Custom collate function for chunked 4D positional data."""
    input_ids_batch = torch.stack([item['input_ids'] for item in batch])
    labels_batch = torch.stack([item['labels'] for item in batch])
    attention_mask_batch = torch.stack([item['attention_mask'] for item in batch])

    return {
        'input_ids': input_ids_batch,
        'attention_mask': None,
        'labels': labels_batch,
    }


def main():
    # Setup paths
    project_dir = Path(__file__).resolve().parents[1]
    data_dir = project_dir / "data" / "aria-midi-v1-unique" / "data"

    # Get all MIDI files
    all_files = list(sorted(data_dir.glob("**/*.mid")))
    print(f"Found {len(all_files)} MIDI files")

    # Split into train/val
    train_files, val_files = train_test_split(all_files, test_size=0.05, random_state=42)
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")

    # Create datasets (no tokenizer needed)
    print("Creating train dataset...")
    train_dataset = MidiDataset4D(train_files[:1], max_seq_len=MAX_SEQ_LEN)

    print("Creating val dataset...")
    val_dataset = MidiDataset4D(train_files[:1], max_seq_len=MAX_SEQ_LEN)
    print(train_files[:1])

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=0
    )

    # Setup logging and checkpoints
    wandb_logger = WandbLogger(project="symbolic-music-new", log_model=True)

    steps_per_epoch = len(train_loader)
    steps_per_half_epoch = steps_per_epoch // 2
    checkpoint_callback = ModelCheckpoint(
        dirpath=project_dir / "checkpoints",
        filename="llama-{epoch:02d}-{step:05d}-{val_loss:.4f}",
        monitor='val_loss',
        save_top_k=5,
        save_last=True,
        mode='min',
        every_n_train_steps=steps_per_half_epoch,
    )

    model = MidiQwenNew(None, train_loader, lr=5e-4, warmup_steps=100)

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        logger=wandb_logger,
        gradient_clip_val=5.0,
        log_every_n_steps=4,
        accelerator="auto",
        callbacks=[checkpoint_callback],
        val_check_interval=0.1,
        precision="bf16-mixed",
        num_sanity_val_steps=0,
    )

    trainer.fit(model, train_loader, val_loader)



if __name__ == "__main__":
    main()