import torch
import torch.nn as nn
import torch.nn.functional as F

class NNLMModel(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int, context_size: int):
        super().__init__()
        # Embedding layer: convert word indices to dense vectors
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        
        # First linear layer: maps concatenated embeddings to hidden layer
        self.linear1 = nn.Linear(embed_size * context_size, hidden_size)
        
        # Second linear layer: maps hidden layer to vocabulary logits
        self.linear2 = nn.Linear(hidden_size, vocab_size)
        
        # Optional bias for direct connection from embeddings to output
        # (used in original NNLM paper to improve performance)
        self.linear_shortcut = nn.Linear(embed_size * context_size, vocab_size)
    
    def forward(self, inputs):
        """
        inputs: Tensor of shape (batch_size, context_size)
        returns: log probabilities over the vocabulary (batch_size, vocab_size)
        """
        # Look up embeddings for each word in the context
        embeds = self.embeddings(inputs)  # (batch_size, context_size, embed_size)
        
        # Flatten (concatenate embeddings of all context words)
        x = embeds.view(embeds.size(0), -1)  # (batch_size, context_size * embed_size)
        
        # Nonlinear hidden layer
        h = torch.tanh(self.linear1(x))  # (batch_size, hidden_size)
        
        # Compute logits (two paths: hidden and direct connection)
        output = self.linear2(h) + self.linear_shortcut(x)  # (batch_size, vocab_size)
        
        # Softmax over vocabulary
        log_probs = F.log_softmax(output, dim=1)
        return log_probs
