"""
MS COCO Dataset Loader for Image Captioning
Handles data loading, preprocessing, and vocabulary building
"""

import os
import json
import pickle
from collections import Counter
from typing import List, Tuple, Dict
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pycocotools.coco import COCO
import nltk
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class Vocabulary:
    """Vocabulary class to handle word-to-index and index-to-word mappings"""
    
    def __init__(self, freq_threshold: int = 5):
        """
        Args:
            freq_threshold: Minimum frequency of a word to be included in vocabulary
        """
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1: "<START>", 2: "<END>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}
        self.word_freq = Counter()
    
    def __len__(self):
        return len(self.itos)
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize text into words"""
        return word_tokenize(text.lower())
    
    def build_vocabulary(self, sentence_list: List[str]):
        """Build vocabulary from list of sentences"""
        frequencies = Counter()
        idx = 4
        
        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1
        
        for word, count in frequencies.items():
            if count >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1
    
    def numericalize(self, text: str) -> List[int]:
        """Convert text to list of indices"""
        tokenized_text = self.tokenize(text)
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]


class CocoDataset(Dataset):
    """MS COCO Dataset for Image Captioning"""
    
    def __init__(
        self,
        root_dir: str,
        annotation_file: str,
        transform=None,
        freq_threshold: int = 5,
        vocab: Vocabulary = None
    ):
        """
        Args:
            root_dir: Root directory containing images
            annotation_file: Path to COCO annotation JSON file
            transform: Image transforms
            freq_threshold: Minimum word frequency for vocabulary
            vocab: Pre-built vocabulary (if None, builds from scratch)
        """
        self.root_dir = root_dir
        self.coco = COCO(annotation_file)
        self.transform = transform
        
        # Get all image IDs and their captions
        self.ids = list(self.coco.anns.keys())
        self.imgs = {}
        self.captions = []
        
        for ann_id in self.ids:
            ann = self.coco.anns[ann_id]
            caption = ann['caption']
            img_id = ann['image_id']
            img_info = self.coco.loadImgs(img_id)[0]
            img_path = os.path.join(root_dir, img_info['file_name'])
            
            self.captions.append(caption)
            self.imgs[ann_id] = img_path
        
        # Build or use provided vocabulary
        if vocab is None:
            self.vocab = Vocabulary(freq_threshold)
            self.vocab.build_vocabulary(self.captions)
        else:
            self.vocab = vocab
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Returns:
            image: Preprocessed image tensor
            caption: Numericalized caption tensor
            caption_length: Length of the caption
        """
        ann_id = self.ids[index]
        caption = self.captions[index]
        img_path = self.imgs[ann_id]
        
        # Load and preprocess image
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        
        # Numericalize caption
        numericalized_caption = [self.vocab.stoi["<START>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<END>"])
        
        caption_tensor = torch.tensor(numericalized_caption, dtype=torch.long)
        
        return image, caption_tensor, len(numericalized_caption)


def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom collate function to handle variable-length captions
    """
    images, captions, caption_lengths = zip(*batch)
    images = torch.stack(images, 0)
    
    # Pad captions to same length
    max_len = max(caption_lengths)
    padded_captions = torch.zeros(len(captions), max_len, dtype=torch.long)
    
    for i, cap in enumerate(captions):
        padded_captions[i, :len(cap)] = cap
    
    caption_lengths = torch.tensor(caption_lengths, dtype=torch.long)
    
    return images, padded_captions, caption_lengths


def get_data_loader(
    root_dir: str,
    annotation_file: str,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    freq_threshold: int = 5,
    vocab: Vocabulary = None
) -> Tuple[DataLoader, Vocabulary]:
    """
    Create DataLoader for COCO dataset
    
    Returns:
        DataLoader and Vocabulary
    """
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # Inception v3 input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    dataset = CocoDataset(
        root_dir=root_dir,
        annotation_file=annotation_file,
        transform=transform,
        freq_threshold=freq_threshold,
        vocab=vocab
    )
    
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return data_loader, dataset.vocab

