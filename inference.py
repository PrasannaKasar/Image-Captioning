"""
Inference Script for Image Captioning Model
Generate captions for new images
"""

import os
import argparse
import torch
from PIL import Image
from torchvision import transforms
import pickle

from model import ImageCaptioningModel
from dataset import Vocabulary


def load_image(image_path: str, device: torch.device) -> torch.Tensor:
    """Load and preprocess image"""
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.to(device)


def generate_caption(
    model: ImageCaptioningModel,
    image_path: str,
    vocab: Vocabulary,
    device: torch.device,
    max_length: int = 20
) -> str:
    """Generate caption for an image"""
    model.eval()
    
    # Load and preprocess image
    image = load_image(image_path, device)
    
    # Generate caption
    with torch.no_grad():
        features = model.encoder(image)
        sampled_ids = model.decoder.sample(features, vocab=vocab)
    
    # Convert indices to words
    caption_words = []
    for idx in sampled_ids:
        if idx in vocab.itos:
            word = vocab.itos[idx]
            if word == "<END>":
                break
            if word not in ["<START>", "<PAD>", "<UNK>"]:
                caption_words.append(word)
    
    # Join words into sentence
    caption = ' '.join(caption_words)
    return caption


def main():
    parser = argparse.ArgumentParser(description='Generate captions for images')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--vocab', type=str, required=True,
                        help='Path to vocabulary pickle file')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--embed_size', type=int, default=256,
                        help='Embedding size (default: 256)')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='LSTM hidden size (default: 512)')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='Number of LSTM layers (default: 1)')
    parser.add_argument('--max_length', type=int, default=20,
                        help='Maximum caption length (default: 20)')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load vocabulary
    print(f'Loading vocabulary from {args.vocab}...')
    with open(args.vocab, 'rb') as f:
        vocab = pickle.load(f)
    print(f'Vocabulary size: {len(vocab)}')
    
    # Load model
    print(f'Loading model from {args.checkpoint}...')
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    model = ImageCaptioningModel(
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        vocab_size=len(vocab),
        num_layers=args.num_layers
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Model loaded successfully!')
    
    # Generate caption
    print(f'\nGenerating caption for {args.image}...')
    caption = generate_caption(
        model, args.image, vocab, device, args.max_length
    )
    
    print(f'\nGenerated Caption: {caption}')


if __name__ == '__main__':
    main()

