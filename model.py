"""
Image Captioning Model
Inception v3 (CNN Encoder) + LSTM (Decoder)
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from typing import Tuple


class EncoderCNN(nn.Module):
    """CNN Encoder using Inception v3"""
    
    def __init__(self, embed_size: int = 256):
        """
        Args:
            embed_size: Size of the embedding vector
        """
        super(EncoderCNN, self).__init__()
        
        # Load pre-trained Inception v3
        try:
            # Try new torchvision API (>=0.13.0)
            inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, aux_logits=False)
        except (AttributeError, TypeError):
            # Fall back to old API
            inception = models.inception_v3(pretrained=True, aux_logits=False)
        
        # Remove the final fully connected layer
        modules = list(inception.children())[:-1]
        self.inception = nn.Sequential(*modules)
        
        # Freeze all parameters except the last few layers
        for param in self.inception.parameters():
            param.requires_grad = False
        
        # Unfreeze the last few layers for fine-tuning
        for param in list(self.inception.children())[-3:]:
            for p in param.parameters():
                p.requires_grad = True
        
        # Add a linear layer to map to embedding size
        # Inception v3 output is 2048 features
        self.linear = nn.Linear(2048, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: Input images tensor (batch_size, 3, 299, 299)
        
        Returns:
            features: Encoded image features (batch_size, embed_size)
        """
        with torch.no_grad():
            # Inception v3 forward pass
            features = self.inception(images)
        
        # Handle the output shape (batch_size, 2048, 1, 1)
        features = features.view(features.size(0), -1)
        
        # Project to embed_size
        features = self.linear(features)
        features = self.bn(features)
        features = self.relu(features)
        features = self.dropout(features)
        
        return features


class DecoderRNN(nn.Module):
    """LSTM Decoder for generating captions"""
    
    def __init__(
        self,
        embed_size: int,
        hidden_size: int,
        vocab_size: int,
        num_layers: int = 1,
        max_seq_length: int = 20
    ):
        """
        Args:
            embed_size: Size of word embeddings
            hidden_size: Size of LSTM hidden state
            vocab_size: Size of vocabulary
            num_layers: Number of LSTM layers
            max_seq_length: Maximum sequence length for generation
        """
        super(DecoderRNN, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.max_seq_length = max_seq_length
        
        # Word embedding layer
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=0.5 if num_layers > 1 else 0
        )
        
        # Output layer
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
    
    def forward(
        self,
        features: torch.Tensor,
        captions: torch.Tensor,
        lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            features: Encoded image features (batch_size, embed_size)
            captions: Caption sequences (batch_size, max_seq_length)
            lengths: Actual lengths of captions (batch_size,)
        
        Returns:
            outputs: Predicted word probabilities (batch_size, max_seq_length, vocab_size)
        """
        # Embed captions
        embeddings = self.embed(captions)
        
        # Concatenate image features with caption embeddings
        # Image features are used as the first input
        features = features.unsqueeze(1)  # (batch_size, 1, embed_size)
        embeddings = torch.cat((features, embeddings), 1)  # (batch_size, max_seq_length+1, embed_size)
        
        # Pack padded sequence for efficient LSTM processing
        packed = pack_padded_sequence(
            embeddings,
            lengths.cpu() + 1,  # +1 for the image feature
            batch_first=True,
            enforce_sorted=False
        )
        
        # LSTM forward pass
        hiddens, _ = self.lstm(packed)
        
        # Unpack sequence
        hiddens, _ = torch.nn.utils.rnn.pad_packed_sequence(
            hiddens,
            batch_first=True
        )
        
        # Remove the first output (corresponding to image features)
        hiddens = hiddens[:, 1:, :]
        
        # Apply dropout and linear layer
        outputs = self.linear(self.dropout(hiddens))
        
        return outputs
    
    def sample(
        self,
        features: torch.Tensor,
        states: Tuple = None,
        vocab: 'Vocabulary' = None
    ) -> List[int]:
        """
        Generate caption using greedy search
        
        Args:
            features: Encoded image features (1, embed_size)
            states: LSTM hidden states
            vocab: Vocabulary object for converting indices to words
        
        Returns:
            sampled_ids: List of word indices
        """
        sampled_ids = []
        inputs = features.unsqueeze(1)  # (1, 1, embed_size)
        
        for i in range(self.max_seq_length):
            hiddens, states = self.lstm(inputs, states)  # hiddens: (1, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))  # (1, vocab_size)
            predicted = outputs.argmax(1)  # (1,)
            sampled_ids.append(predicted.item())
            
            # Stop if <END> token is generated
            if vocab is not None and predicted.item() == vocab.stoi["<END>"]:
                break
            
            # Use predicted word as next input
            inputs = self.embed(predicted).unsqueeze(1)  # (1, 1, embed_size)
        
        return sampled_ids


class ImageCaptioningModel(nn.Module):
    """Complete Image Captioning Model: CNN Encoder + LSTM Decoder"""
    
    def __init__(
        self,
        embed_size: int = 256,
        hidden_size: int = 512,
        vocab_size: int = 10000,
        num_layers: int = 1,
        max_seq_length: int = 20
    ):
        """
        Args:
            embed_size: Size of embeddings
            hidden_size: Size of LSTM hidden state
            vocab_size: Size of vocabulary
            num_layers: Number of LSTM layers
            max_seq_length: Maximum sequence length
        """
        super(ImageCaptioningModel, self).__init__()
        
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(
            embed_size,
            hidden_size,
            vocab_size,
            num_layers,
            max_seq_length
        )
    
    def forward(
        self,
        images: torch.Tensor,
        captions: torch.Tensor,
        lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            images: Input images (batch_size, 3, 299, 299)
            captions: Caption sequences (batch_size, max_seq_length)
            lengths: Caption lengths (batch_size,)
        
        Returns:
            outputs: Predicted word probabilities
        """
        features = self.encoder(images)
        outputs = self.decoder(features, captions, lengths)
        return outputs
    
    def generate_caption(
        self,
        image: torch.Tensor,
        vocab: 'Vocabulary',
        max_length: int = 20
    ) -> List[str]:
        """
        Generate caption for a single image
        
        Args:
            image: Input image tensor (1, 3, 299, 299)
            vocab: Vocabulary object
            max_length: Maximum caption length
        
        Returns:
            List of words in the generated caption
        """
        self.eval()
        with torch.no_grad():
            features = self.encoder(image)
            sampled_ids = self.decoder.sample(features, vocab=vocab)
        
        # Convert indices to words
        caption = []
        for idx in sampled_ids:
            if idx in vocab.itos:
                word = vocab.itos[idx]
                if word == "<END>":
                    break
                if word not in ["<START>", "<PAD>", "<UNK>"]:
                    caption.append(word)
        
        return caption

