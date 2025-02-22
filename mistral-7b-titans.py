import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
from transformers import MistralForCausalLM, AutoTokenizer

# Check for required dependencies
try:
    import sentencepiece
    import google.protobuf
except ImportError as e:
    raise ImportError(
        "Missing required dependencies. Please install them with:\n"
        "pip install sentencepiece protobuf\n"
        "Then restart your Python session."
    ) from e

# Hyperparameters
memory_size = 20
beta = 0.1
eta = 0.01
theta = 0.1
max_seq_len = 256
learning_rate = 0.001
temperature = 0.7
top_k = 50
top_p = 0.9
num_repetitions = 10

# Device setup for multi-platform support
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device (Apple Silicon).")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("No GPU available. Falling back to CPU (may be slow for Mistral 7B).")

class TitansMistral(nn.Module):
    def __init__(self, base_model, memory_size=memory_size):
        super(TitansMistral, self).__init__()
        self.base_model = base_model.to(device)
        self.d_model = base_model.config.hidden_size
        self.memory_weight = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.ReLU(),
            nn.Linear(self.d_model * 2, self.d_model * 2),
            nn.ReLU(),
            nn.Linear(self.d_model * 2, self.d_model)
        ).to(device)
        self.register_buffer('memory', torch.zeros(1, memory_size, self.d_model).to(device))
        self.register_buffer('previous_surprise', torch.zeros(1, memory_size, self.d_model).to(device))

    def get_positional_encoding(self, seq_len, d_model):
        position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / d_model))
        pos_encoding = torch.zeros(seq_len, d_model, device=device)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding.unsqueeze(0)

    def forward(self, input_ids=None, inputs_embeds=None, memory=None, update_memory=True, repetition=0):
        if input_ids is not None and inputs_embeds is None:
            inputs_embeds = self.base_model.model.embed_tokens(input_ids).to(device)
        if memory is None:
            memory = self.memory.expand(inputs_embeds.size(0), memory_size, self.d_model).to(device)

        pos_encoding = self.get_positional_encoding(memory_size, self.d_model).to(device)
        memory = memory + pos_encoding[:, :memory_size, :]

        memory_scale = 5 + repetition
        x_with_memory = torch.cat([inputs_embeds, memory * memory_scale], dim=1).to(device)

        outputs = self.base_model(inputs_embeds=x_with_memory)
        logits = outputs.logits[:, :inputs_embeds.size(1), :].to(device)

        if update_memory:
            surprise = self.memory_weight(inputs_embeds.mean(dim=1).unsqueeze(1)).to(device)
            new_memory = torch.cat([memory[:, 1:, :], surprise], dim=1).detach().to(device)
            if inputs_embeds.size(0) == 1:
                self.memory = new_memory
            return logits, new_memory
        return logits

def apply_repetition_penalty(logits, generated, penalty=50.0, n=8):
    for i in range(max(0, generated.size(1) - n), generated.size(1)):
        ngram = tuple(generated[0, i:i+n].tolist())
        if ngram in [tuple(generated[0, j:j+n].tolist()) for j in range(max(0, i-n))]:
            for token in ngram:
                logits[0, token] /= penalty
    return logits

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0):
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[..., indices_to_remove] = float('-inf')
    return logits

def generate_and_learn(model, tokenizer, start_text, new_context=None, max_length=50, temperature=temperature, update_memory=True, repetition=0):
    model.eval()
    device = model.base_model.device
    start_ids = tokenizer.encode(start_text, return_tensors="pt").to(device)
    if new_context:
        context_ids = tokenizer.encode(new_context, return_tensors="pt").to(device)
        input_ids = torch.cat([start_ids, context_ids], dim=1)
    else:
        input_ids = start_ids
    generated = input_ids.clone()
    optimizer = torch.optim.Adam(model.memory_weight.parameters(), lr=learning_rate) if update_memory else None

    memory_before = model.memory.clone().detach().to(device)
    if new_context and update_memory:
        context_ids = tokenizer.encode(new_context, return_tensors="pt").to(device)
        model.train()
        optimizer.zero_grad()

        inputs_embeds = model.base_model.model.embed_tokens(context_ids).to(device)
        inputs_embeds.requires_grad_(True)
        memory = model.memory.clone().detach().requires_grad_(True).to(device)
        memory_scale = 5 + repetition
        x_with_memory = torch.cat([inputs_embeds, memory * memory_scale], dim=1).to(device)

        outputs = model.base_model(inputs_embeds=x_with_memory)
        logits = outputs.logits[:, :inputs_embeds.size(1), :]
        loss = F.cross_entropy(logits.view(-1, model.base_model.config.vocab_size), context_ids.view(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        print(f"Loss: {loss.item():.4f}")

        grad = inputs_embeds.grad if inputs_embeds.grad is not None else torch.zeros_like(inputs_embeds)
        dynamic_theta = theta * min(2.0, 1 + math.log(repetition + 1))
        S_t = eta * model.previous_surprise[:, :grad.size(1), :] - dynamic_theta * grad
        model.previous_surprise = F.pad(S_t, (0, 0, 0, memory_size - S_t.size(1)), "constant", 0).detach().to(device)

        alpha_t = min(beta * loss.item() * (repetition + 1), 0.5)
        new_memory = model.memory.clone()
        update_slots = min(grad.size(1), memory_size)
        new_memory[:, -update_slots:, :] = ((1 - alpha_t) * new_memory[:, -update_slots:, :] + alpha_t * S_t).detach()
        model.memory = new_memory.to(device)

        optimizer.step()
        model.eval()
        print(f"Memory updated with new context: {new_context}")

    with torch.no_grad():
        memory = model.memory.to(device)
        for _ in range(max_length):
            result = model(input_ids=generated, memory=memory, update_memory=update_memory and new_context is not None, repetition=repetition)
            if update_memory and new_context is not None:
                logits, memory = result
                model.memory = memory.to(device)
            else:
                logits = result
            logits = logits[:, -1, :].to(device)
            logits = apply_repetition_penalty(logits, generated, penalty=50.0, n=8)
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1).to(device)
            if generated.size(1) > max_seq_len:
                generated = generated[:, -max_seq_len:]

        memory_after = model.memory.clone().detach().to(device)
        memory_changed = not torch.allclose(memory_before, memory_after, atol=1e-5)
        print(f"Memory changed: {memory_changed}")
        if memory_changed:
            print(f"Memory diff norm: {torch.norm(memory_after - memory_before):.4f}")
        else:
            print("Memory unchanged - check update logic")

        generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        context_words = ["your", "helpful", "assistant"] if new_context else []
        for word in context_words:
            if word in generated_text.lower():
                print(f"Found context word '{word}' in output")

        return generated_text

def interactive_session(model, tokenizer):
    weights_path = f"titans_mistral_{device.type}.pth"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Loaded saved weights from '{weights_path}'")
    else:
        print("No saved weights found. Using pre-trained Mistral 7B weights.")

    sample_text = "Hello, this is a test."
    encoded = tokenizer.encode(sample_text, return_tensors="pt").to(device)
    decoded = tokenizer.decode(encoded[0], skip_special_tokens=True)
    print(f"Tokenizer test - Encoded: {encoded}, Decoded: {decoded}")

    print("\nStarting learning test...")
    start_text = "I am Penny"
    new_context = " your helpful assistant"

    print("\nControl: No memory updates")
    control_text = generate_and_learn(model, tokenizer, start_text, new_context=None, update_memory=False, repetition=0)
    print(f"Generated Text (no updates):\n{control_text}\n")

    print(f"Testing learning with {num_repetitions} repetitions of context: '{new_context}'")
    for i in range(num_repetitions):
        print(f"\nRepetition {i+1}:")
        generated_text = generate_and_learn(model, tokenizer, start_text, new_context=new_context, repetition=i)
        print(f"Generated Text:\n{generated_text}\n")

    print("Entering manual mode...")
    while True:
        start_text = input("Enter starting text (or 'quit' to exit): ")
        if start_text.lower() == "quit":
            torch.save({
                'memory_weight': model.memory_weight.state_dict(),
                'memory': model.memory,
                'previous_surprise': model.previous_surprise
            }, weights_path)
            print(f"Memory-related weights saved to '{weights_path}'. Exiting.")
            break
        new_context = input("Enter new context to learn (or press Enter to skip): ")
        if not new_context:
            new_context = None
        generated_text = generate_and_learn(model, tokenizer, start_text, new_context=new_context, repetition=0)
        print(f"Generated Text:\n{generated_text}\n")

if __name__ == "__main__":
    try:
        base_model = MistralForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", torch_dtype=torch.float16).to(device)
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        print("Ensure 'sentencepiece' and 'protobuf' are installed and try again.")
        exit(1)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = TitansMistral(base_model, memory_size=memory_size).to(device)
    print(f"Model moved to {device}. Ready to run.")
    
    if device.type == "cuda":
        print(f"CUDA memory summary:\n{torch.cuda.memory_summary(device=device)}")
    
    interactive_session(model, tokenizer)
