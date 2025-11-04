# LCT Tools

Utility scripts for LLM Comparative Tool.

## 🔍 Model Checking Tools

### Gated Model Checker

Check if HuggingFace models require approval before using them in experiments.

**Usage:**

```bash
# Quick check with script
./check_models.sh gpt2 meta-llama/Llama-2-7b-hf

# Or use Python directly
.venv/bin/python app/src/tools/gated_model_checker.py meta-llama/Llama-2-7b-hf

# Check multiple models
.venv/bin/python app/src/tools/gated_model_checker.py \
  --check-list gpt2 microsoft/phi-2 meta-llama/Llama-2-7b-hf \
  --alternatives

# Check from file
.venv/bin/python app/src/tools/gated_model_checker.py \
  --batch-file recommended_open_models.txt
```

**Output:**
- ✅ Open models (ready to use)
- 🔒 Gated models (need approval)
- 📂 Suggested alternatives

### Model Compatibility Checker

Verify if a model is compatible with the LCT system.

**Usage:**

```bash
.venv/bin/python app/src/tools/model_compatibility_checker.py gpt2
.venv/bin/python app/src/tools/model_compatibility_checker.py --list-recommended
```

## 📋 Recommended Models

See `recommended_open_models.txt` for a curated list of open (non-gated) models.

**Quick thesis setup (10 models, all open):**
- gpt2, gpt2-medium
- google/flan-t5-base, google/flan-t5-large
- EleutherAI/pythia-410m, EleutherAI/pythia-1.4b
- microsoft/phi-1_5, microsoft/phi-2
- facebook/opt-350m, bigscience/bloomz-560m

## 🔐 HuggingFace Authentication

Some models require authentication:

```bash
# Get your token from: https://huggingface.co/settings/tokens
export HF_TOKEN="hf_your_token_here"

# Add to ~/.bashrc for permanent use
echo 'export HF_TOKEN="hf_your_token_here"' >> ~/.bashrc
```

## 📖 Full Documentation

See `GATED_MODELS_SOLUTION.md` for comprehensive guide on:
- What are gated models
- How to get access
- Open alternatives
- Decision tree for model selection
