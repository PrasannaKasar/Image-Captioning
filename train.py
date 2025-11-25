"""
Training Script for Image Captioning Model
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pickle

from model import ImageCaptioningModel
from dataset import get_data_loader, Vocabulary


def train(
    train_loader,
    model,
    criterion,
    optimizer,
    epoch,
    device,
    writer=None
):
    """Train the model for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for images, captions, lengths in pbar:
        images = images.to(device)
        captions = captions.to(device)
        lengths = lengths.to(device)
        
        # Forward pass
        outputs = model(images, captions, lengths)
        
        # Remove <START> token from captions for loss calculation
        targets = captions[:, 1:]
        
        # Reshape outputs and targets for loss calculation
        outputs = outputs.reshape(-1, outputs.size(2))
        targets = targets.reshape(-1)
        
        # Calculate loss
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / num_batches
    
    if writer:
        writer.add_scalar('Loss/Train', avg_loss, epoch)
    
    return avg_loss


def validate(val_loader, model, criterion, device, epoch, writer=None):
    """Validate the model"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for images, captions, lengths in tqdm(val_loader, desc='Validation'):
            images = images.to(device)
            captions = captions.to(device)
            lengths = lengths.to(device)
            
            outputs = model(images, captions, lengths)
            targets = captions[:, 1:]
            
            outputs = outputs.reshape(-1, outputs.size(2))
            targets = targets.reshape(-1)
            
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    
    if writer:
        writer.add_scalar('Loss/Validation', avg_loss, epoch)
    
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description='Train Image Captioning Model')
    parser.add_argument('--train_dir', type=str, required=True,
                        help='Directory containing training images')
    parser.add_argument('--train_annotations', type=str, required=True,
                        help='Path to training annotations JSON file')
    parser.add_argument('--val_dir', type=str, required=True,
                        help='Directory containing validation images')
    parser.add_argument('--val_annotations', type=str, required=True,
                        help='Path to validation annotations JSON file')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of epochs (default: 10)')
    parser.add_argument('--embed_size', type=int, default=256,
                        help='Embedding size (default: 256)')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='LSTM hidden size (default: 512)')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='Number of LSTM layers (default: 1)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--freq_threshold', type=int, default=5,
                        help='Minimum word frequency for vocabulary (default: 5)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers (default: 4)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints (default: ./checkpoints)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load training data
    print('Loading training data...')
    train_loader, vocab = get_data_loader(
        root_dir=args.train_dir,
        annotation_file=args.train_annotations,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        freq_threshold=args.freq_threshold
    )
    
    # Save vocabulary
    vocab_path = os.path.join(args.save_dir, 'vocab.pkl')
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print(f'Vocabulary saved to {vocab_path}')
    print(f'Vocabulary size: {len(vocab)}')
    
    # Load validation data (use same vocabulary)
    print('Loading validation data...')
    val_loader, _ = get_data_loader(
        root_dir=args.val_dir,
        annotation_file=args.val_annotations,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        vocab=vocab
    )
    
    # Initialize model
    model = ImageCaptioningModel(
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        vocab_size=len(vocab),
        num_layers=args.num_layers
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore <PAD> tokens
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'logs'))
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        print(f'Resuming from checkpoint: {args.resume}')
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))
    
    # Training loop
    print('Starting training...')
    for epoch in range(start_epoch, args.num_epochs):
        print(f'\nEpoch {epoch+1}/{args.num_epochs}')
        
        # Train
        train_loss = train(
            train_loader, model, criterion, optimizer, epoch, device, writer
        )
        print(f'Train Loss: {train_loss:.4f}')
        
        # Validate
        val_loss = validate(
            val_loader, model, criterion, device, epoch, writer
        )
        print(f'Validation Loss: {val_loss:.4f}')
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'vocab_size': len(vocab),
            'embed_size': args.embed_size,
            'hidden_size': args.hidden_size,
        }
        
        # Save latest checkpoint
        latest_path = os.path.join(args.save_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.save_dir, 'best_checkpoint.pth')
            torch.save(checkpoint, best_path)
            print(f'New best model saved! Validation Loss: {val_loss:.4f}')
    
    print('\nTraining completed!')
    writer.close()


if __name__ == '__main__':
    main()

