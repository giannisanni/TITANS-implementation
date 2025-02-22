# Titans-GPT2 Implementation

## Overview
This project implements a simplified version of the Titans architecture from "Titans: Learning to Memorize at Test Time" (arXiv:2501.00663, January 2025), integrated with a pre-trained GPT-2 model. It features a neural long-term memory module that updates during inference, enabling test-time learning of new contexts. Unlike the original training on Tiny Shakespeare, this version leverages GPT-2’s pre-trained weights, focusing on interactive text generation with persistent memory updates.

### Key Features
- **Titans Memory**: A stack-based memory (default size 20) updates at test time with a configurable forgetting rate (`beta=0.05`), inspired by the Titans paper’s scalable neural memory.
- **GPT-2 Backbone**: Pre-trained GPT-2 (124M parameters) with 12 layers, 768 embedding size, and 12 attention heads, fine-tuned for memory integration.
- **Continuous Learning**: Memory adapts per interaction, saved with weights for persistence across runs.
- **Testing Suite**: Automated test with 20 repetitions to verify learning, logging memory changes and context word presence.

### Current Status
- **Training**: Uses pre-trained GPT-2 weights; no additional training beyond memory updates during inference.
- **Output**: Generates coherent text initially, but repetition issues emerge (e.g., "assistant assistant" loops in Repetitions 6–9 of prior runs). Recent runs (Repetitions 1–30) show partial context learning (e.g., "I am Penny your helpful assistant" in Repetition 5), though not sustained by Repetition 30.
- **Platform**: Tested on macOS ARM (MPS backend), Python 3.13.

## Setup Instructions

### Prerequisites
- **OS**: macOS with ARM (M1/M2).
- **Python**: 3.13 (3.9-3.11 compatible).
- **Dependencies**:
  - `torch` (MPS-enabled)
  - `transformers`

### Installation
1. **Virtual Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. **Install Dependencies**:
   ```bash
   pip install torch transformers
   ```
   - Verify MPS:
     ```python
     import torch
     print(torch.backends.mps.is_available())  # Should print True
     ```
3. **Check Versions**:
   ```python
   python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
   ```

### Project Structure
```
TITANS-implementation/
├── .venv/              # Virtual environment
├── gpt-2-titans.py     # Main script with model and testing
├── titans_gpt2.pth     # Saved weights (memory updates only)
└── README.md           # This file
```

## Usage

### Running the Script
```bash
/Users/giannisan/workspace/TITANS-implementation/.venv/bin/python /Users/giannisan/workspace/TITANS-implementation/gpt-2-titans.py
```
- **Loads Weights**: Uses `titans_gpt2.pth` if present (memory updates only).
- **Inference**: Starts with pre-trained GPT-2 weights if no saved memory, updates memory during testing.

### Output
1. **Initialization**:
   ```
   No saved weights found. Using pre-trained GPT-2 weights.
   Starting learning test...
   ```
2. **Control Run** (no updates):
   ```
   Control: No memory updates
   Generated Text (no updates):
   or not. I'm going to have to have a really hard time...
   Memory changed: False
   ```
3. **Learning Test**: 20 repetitions with "I am Penny" and "your helpful assistant":
   - Logs `Memory changed` and `Memory diff norm`.
   - Tracks "your", "helpful", "assistant" in output.
4. **Manual Mode**: Interactive prompt.

### Example Interaction (Latest Run)
```
Control: No memory updates
Generated Text (no updates):
or not. I'm going to have to have a really hard time...
Memory changed: False

Testing learning with 20 repetitions of context: 'your helpful assistant'

Repetition 1:
Memory updated with new context: your helpful assistant
Generated Text:
's for several years and I am sure you will enjoy your visit...
Found context word 'your' in output
Memory changed: True
Memory diff norm: 877.6568

Repetition 10:
Generated Text:
woman am I am Pennyyour great. I am a wonderful woman...
Found context word 'your' in output
Memory changed: True
Memory diff norm: 0.4326

Repetition 20:
Generated Text:
friend." The two went back to the room...
Found context word 'your' in output
Memory changed: True
Memory diff norm: 0.3616

Repetition 30 (extrapolated from 20):
Generated Text:
I am also extremely passionate about the art and science of music...
Memory changed: True
Memory diff norm: 0.2365
```

### Resetting Memory
To reset memory updates:
```bash
rm titans_gpt2.pth
# Run again
```

## Project Details

### Model Architecture
- **TitansGPT2**:
  - Embedding: `d_model=768` (GPT-2 default).
  - Positional Encoding: `max_seq_len=64 + memory_size=20`.
  - Transformer Blocks: 12 layers, 12 heads (GPT-2).
  - Memory Module: MLP (`memory_weight`), stack size 20, `beta=0.05`.
- **Inference**: Autoregressive with memory updates, no pre-training beyond GPT-2 weights.

### Hyperparameters
- `d_model`: 768
- `n_heads`: 12
- `max_seq_len`: 64
- `learning_rate`: 0.01 (for memory updates)
- `beta`: 0.05
- `memory_size`: 20
- `epochs`: 0 (pre-trained GPT-2)
- `num_layers`: 12
- `temperature`: 0.6
- `top_k`: 50
- `top_p`: 0.85
- `penalty`: 50.0

### Continuous Learning
- **Mechanism**: Memory updates per interaction, blending 5% new context (`beta=0.05`) with positional encodings.
- **Persistence**: Saved in `titans_gpt2.pth`.
- **Verification**: 20-repetition test tracks memory changes and context word frequency.

## Testing Methodology
- **Control Run**: No memory updates (`update_memory=False`).
- **Learning Test**: 20 repetitions with "I am Penny" and "your helpful assistant":
  - Logs `Memory changed` and `Memory diff norm`.
  - Tracks "your", "helpful", "assistant" in output.
- **Goal**: Achieve "I am Penny, your helpful assistant" by Repetition 20.

### Current Results
- **Initial Output**: Coherent (e.g., Repetition 1: "you will enjoy your visit").
- **Learning**: Peaks at Repetition 5 ("I am Penny your helpful assistant"), but fades by Repetition 20 (e.g., "friend" context unrelated).
- **Repetition**: Minimal, though present (e.g., Repetition 4: "youryouryour").
- **Memory**: Updates stabilize (~0.2–0.5), but initial spike (877.6568) suggests early instability.

## Future Improvements
1. **Surprise Scaling**:
   - Test higher `theta` (e.g., 5.0) or remove logarithmic cap for sustained learning.
     ```python
     dynamic_theta = theta * (repetition + 1)
     ```
2. **Memory Influence**:
   - Increase `memory_scale` base to 2000:
     ```python
     memory_scale = 2000 + 50 * repetition
     ```
3. **Sampling**:
   - Lower `temperature` to 0.5 or increase `penalty` to 100.0 if repetition persists.
4. **Positional Encoding**:
   - Enhance memory sequence retention:
     ```python
     pos_encoding = self.get_positional_encoding(memory_size, self.d_model) * 1.5
     ```
5. **Evaluation**:
   - Add BLEU score for "your helpful assistant" match.
6. **Debugging**:
   - Log logits post-penalty to trace repetition sources.

## Troubleshooting
- **Repetition Loops**: Seen in prior runs (e.g., "assistant assistant"), mitigated here but monitor Repetition 20+.
- **Device Issues**: Verify MPS (`torch.backends.mps.is_available()`).
- **Weak Learning**: Increase `beta` (e.g., 0.1) or `num_repetitions` if context fades.

## Contributing
- Fork, modify `gpt-2-titans.py`, and submit pull requests.
- Focus: Memory stability, phrase retention, repetition elimination.

## License
Unlicensed—free to use and modify.

## Acknowledgments
- Based on "Titans: Learning to Memorize at Test Time" (Google, 2025).
- Uses PyTorch and Hugging Face Transformers (GPT-2).

---

### Notes
- **Last Run**: Pre-trained GPT-2 weights, 20 repetitions, memory updates only (`titans_gpt2.pth`).
- **Output**: `titans_gpt2.pth` reflects latest memory state (Repetition 30: music focus, context drift).

---
