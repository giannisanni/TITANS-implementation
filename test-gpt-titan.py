import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
import os

# Hyperparameters
d_model = 256
n_heads = 8
d_ff = 1024
max_seq_len = 64
batch_size = 32
learning_rate = 0.001
memory_dim = d_model
beta = 0.7  # Increased for faster learning
memory_size = 10
epochs = 50
num_layers = 4

# Load Tiny Shakespeare
def load_tiny_shakespeare():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    response = requests.get(url)
    text = response.text
    return text

# Character-level tokenizer
class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text):
        text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
        return [self.char_to_idx[ch] for ch in text]

    def decode(self, indices):
        return ''.join([self.idx_to_char[idx] for idx in indices])

# Prepare data
def prepare_data(text, seq_len):
    tokenizer = CharTokenizer(text)
    encoded = tokenizer.encode(text)
    data = torch.tensor(encoded, dtype=torch.long)
    n_sequences = len(data) // (seq_len + 1)
    data = data[:n_sequences * (seq_len + 1)].view(n_sequences, seq_len + 1)
    return data, tokenizer

# Multi-Head Self-Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size = x.size(0)
        query = self.q_linear(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        key = self.k_linear(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        value = self.v_linear(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, value).transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.n_heads * self.d_k)
        return self.out_linear(context)

# Feed-Forward Network
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

# Transformer Layer
class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super(TransformerLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        attn_out = self.attention(x, mask)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x

# Titans-GPT2 Model
class TitansGPT2(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, max_seq_len, num_layers, memory_size):
        super(TitansGPT2, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_seq_len + memory_size, d_model))
        self.layers = nn.ModuleList([TransformerLayer(d_model, n_heads, d_ff) for _ in range(num_layers)])
        self.output = nn.Linear(d_model, vocab_size)
        
        self.memory_weight = nn.Linear(d_model, d_model, bias=False)
        self.memory = torch.zeros(1, memory_size, d_model)
        self.memory_size = memory_size

    def to(self, device):
        super(TitansGPT2, self).to(device)
        self.memory = self.memory.to(device)
        return self

    def forward(self, x, memory=None, update_memory=True):
        batch_size, seq_len = x.size()
        x = self.embedding(x) + self.pos_encoding[:, :seq_len, :]
        
        if memory is None:
            memory = self.memory.expand(batch_size, self.memory_size, d_model).to(x.device)
        
        x_with_memory = torch.cat([x, memory], dim=1)
        mask = torch.tril(torch.ones(seq_len + self.memory_size, seq_len + self.memory_size)).unsqueeze(0).unsqueeze(0).to(x.device)

        for layer in self.layers:
            x_with_memory = layer(x_with_memory, mask)
        
        logits = self.output(x_with_memory[:, :seq_len, :])
        
        if update_memory:
            last_input = x_with_memory[:, seq_len - 1, :]
            surprise = self.memory_weight(last_input.unsqueeze(1))
            new_memory = torch.cat([memory[:, 1:, :], surprise], dim=1)
            if batch_size == 1:
                self.memory = ((1 - beta) * self.memory + beta * new_memory).detach()
            return logits, new_memory
        return logits

# Training Loop
def train(model, data, tokenizer, epochs=epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in range(0, len(data), batch_size):
            batch_data = data[batch:batch + batch_size].to(device)
            optimizer.zero_grad()
            logits, _ = model(batch_data[:, :-1], update_memory=True)
            loss = criterion(logits.view(-1, tokenizer.vocab_size), batch_data[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / (len(data) // batch_size):.4f}")
    
    torch.save(model.state_dict(), "titans_model.pth")
    print("Model weights saved to 'titans_model.pth'")

# Inference with Continuous Learning and Memory Logging
def generate_and_learn(model, tokenizer, start_text, new_context=None, max_length=100, temperature=0.5, update_memory=True):
    model.eval()
    input_ids = torch.tensor([tokenizer.encode(start_text)], dtype=torch.long).to(device)
    generated = tokenizer.encode(start_text)
    
    with torch.no_grad():
        memory = None
        memory_before = model.memory.clone()
        for _ in range(max_length):
            result = model(input_ids, memory=memory, update_memory=update_memory)
            if update_memory:
                logits, memory = result
            else:
                logits = result
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)
            input_ids = torch.tensor([generated], dtype=torch.long).to(device)
            if input_ids.size(1) > max_seq_len:
                input_ids = input_ids[:, -max_seq_len:]
    
    if new_context and update_memory:
        context_ids = torch.tensor([tokenizer.encode(new_context)], dtype=torch.long).to(device)
        _, memory = model(context_ids, memory=memory, update_memory=True)
        print(f"Memory updated with new context: {new_context}")
    
    memory_after = model.memory.clone()
    memory_changed = not torch.allclose(memory_before, memory_after, atol=1e-5)
    print(f"Memory changed: {memory_changed}")
    if memory_changed:
        print(f"Memory diff norm: {torch.norm(memory_after - memory_before):.4f}")
    
    generated_text = tokenizer.decode(generated)
    context_words = ["that", "is", "the", "question"] if new_context else []
    for word in context_words:
        if word in generated_text.lower():
            print(f"Found context word '{word}' in output")
    
    return generated_text

# Interactive Session with Learning Test
def interactive_session(model, tokenizer):
    if os.path.exists("titans_model.pth"):
        model.load_state_dict(torch.load("titans_model.pth"))
        print("Loaded saved weights from 'titans_model.pth'")
    else:
        print("No saved weights found. Training from scratch...")
        text = load_tiny_shakespeare()
        data, tokenizer = prepare_data(text, max_seq_len)
        data = data.to(device)
        train(model, data, tokenizer)
    
    print("\nStarting learning test...")
    
    # Test parameters
    start_text = "to be or not to be"
    new_context = "that is the question"
    num_repetitions = 20
    
    # Control run: No memory updates
    print("\nControl: No memory updates")
    control_text = generate_and_learn(model, tokenizer, start_text, new_context=None, update_memory=False)
    print(f"Generated Text (no updates):\n{control_text}\n")
    
    # Learning test: Repeated context updates
    print(f"Testing learning with {num_repetitions} repetitions of context: '{new_context}'")
    for i in range(num_repetitions):
        print(f"\nRepetition {i+1}:")
        generated_text = generate_and_learn(model, tokenizer, start_text, new_context=new_context, update_memory=True)
        print(f"Generated Text:\n{generated_text}\n")
    
    # Manual interaction mode
    print("Entering manual mode...")
    while True:
        start_text = input("Enter starting text (or 'quit' to exit): ")
        if start_text.lower() == "quit":
            torch.save(model.state_dict(), "titans_model.pth")
            print("Model weights saved. Exiting.")
            break
        
        new_context = input("Enter new context to learn (or press Enter to skip): ")
        if not new_context:
            new_context = None
        
        generated_text = generate_and_learn(model, tokenizer, start_text, new_context=new_context)
        print(f"Generated Text:\n{generated_text}\n")

# Main Execution
if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    text = load_tiny_shakespeare()
    data, tokenizer = prepare_data(text, max_seq_len)
    model = TitansGPT2(tokenizer.vocab_size, d_model, n_heads, d_ff, max_seq_len, num_layers, memory_size).to(device)
    interactive_session(model, tokenizer)
