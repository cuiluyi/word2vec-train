from collections import Counter
from tqdm import tqdm
from typing import Literal

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger

from NNLM import NNLMModel


def train_word_vectors(
    config: dict,
    text: str,
    min_freq: int = 3,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Split text into words
    words = text.split()

    # Build vocabulary (filter words with frequency < min_freq)
    word_counts = Counter(words)
    vocab = [
        word for word, count in word_counts.most_common() if count >= min_freq
    ]
    vocab_size = len(vocab)
    word_to_ix = {word: i for i, word in enumerate(vocab)}
    ix_to_word = {i: word for word, i in word_to_ix.items()}

    # tokenize data into contexts and targets
    data_contexts = []
    data_targets = []
    for i in tqdm(range(config.context_size, len(words))):
        context = [
            word_to_ix[words[j]]
            for j in range(i - config.context_size, i)
            if words[j] in word_to_ix
        ]
        if len(context) == config.context_size and words[i] in word_to_ix:
            target = word_to_ix[words[i]]
            data_contexts.append(context)
            data_targets.append(target)

    # prepare training dataset
    dataset = TensorDataset(
        torch.tensor(data_contexts, dtype=torch.long),
        torch.tensor(data_targets, dtype=torch.long),
    )
    train_loader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True
    )
    n_iterations = len(train_loader)

    # prepare model, loss function, optimizer
    model = NNLMModel(
        vocab_size,
        config.embed_size,
        config.hidden_size,
        config.context_size,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)

    # training loop
    for epoch in range(config.epochs):
        for i, (context, target) in enumerate(train_loader):
            context = context.to(device)
            target = target.to(device)
            
            # Forward pass
            outputs = model(context)
            loss = criterion(outputs, target)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % config.logging_steps == 0:
                logger.info(
                    f"Epoch [{epoch+1}/{config.epochs}], Step [{i+1}/{n_iterations}], Loss: {loss.item():.4f}"
                )

    # extract word embeddings
    embeddings = model.embeddings.weight.data.cpu().numpy()

    # generate word vector file content (Word2Vec format)
    vector_content = f"{vocab_size} {config.embed_size}\n"
    for i in range(vocab_size):
        word = ix_to_word[i]
        vec = " ".join([f"{v:.6f}" for v in embeddings[i]])
        vector_content += f"{word} {vec}\n"

    return vector_content