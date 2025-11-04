# Gated Models Issue & Solution

## Problem
Some HuggingFace models are **gated** - they require:
1. User authentication (HuggingFace token)
2. Accepting terms of use
3. Sometimes **manual human approval** (can take hours/days)

**Examples from your experiment:**
- ❌ `meta-llama/Llama-3.2-1B-Instruct` - GATED (needs approval)
- ❌ `meta-llama/Llama-2-7b-hf` - GATED (needs approval)
- ❌ `mistralai/Mistral-7B-v0.1` - GATED (needs approval)
- ❌ `google/gemma-2b` - GATED (needs approval)

## Solution: Pre-Check Models Before Experiments

### **Tool 1: Gated Model Checker**

Check if models are gated BEFORE adding them to your experiment:

```bash
# Check single model
./check_models.sh meta-llama/Llama-3.2-1B-Instruct

# Check multiple models
./check_models.sh gpt2 meta-llama/Llama-2-7b-hf mistralai/Mistral-7B-v0.1

# Check from file
echo "meta-llama/Llama-3.2-1B-Instruct" > models_to_check.txt
echo "google-bert/bert-base-uncased" >> models_to_check.txt
./check_models.sh
```

**Output:**
```
✅ OPEN ACCESS (2 models):
   • google-bert/bert-base-uncased
   • openai-community/gpt2

🔒 GATED - ACCESS REQUIRED (1 models):
   ❌ meta-llama/Llama-3.2-1B-Instruct
      → Manual approval needed
      → Visit: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct

📂 ALWAYS OPEN - NO APPROVAL NEEDED:
   • gpt2
   • gpt2-medium
   • EleutherAI/pythia-410m
   • facebook/opt-350m
   ...
```

### **Tool 2: Python API**

Use directly in Python:

```python
from app.src.tools.gated_model_checker import check_model_access

# Check single model
info = check_model_access("meta-llama/Llama-2-7b-hf")

if info.is_gated and not info.has_access:
    print(f"⚠️  {info.model_id} requires approval!")
    print(f"Visit: https://huggingface.co/{info.model_id}")
else:
    print(f"✅ {info.model_id} is ready to use!")
```

## Recommended Open Models for Thesis

### ✅ **Always Open (No Approval)**

**Small Models (Fast, Easy)**
```
gpt2                              # 124M - Classic baseline
gpt2-medium                        # 355M - Better quality
distilgpt2                         # 82M - Faster GPT-2
microsoft/phi-1_5                  # 1.3B - Code/reasoning
EleutherAI/pythia-410m             # 410M - Research focused
EleutherAI/pythia-1.4b             # 1.4B - Better quality
```

**Medium Models (Balanced)**
```
google/flan-t5-base                # 250M - Instruction-tuned
google/flan-t5-large               # 780M - Better T5
facebook/opt-350m                  # 350M - Meta's open model
facebook/opt-1.3b                  # 1.3B - Larger OPT
bigscience/bloomz-560m             # 560M - Multilingual
```

**Larger Models (Quality)**
```
microsoft/phi-2                    # 2.7B - State-of-art small
EleutherAI/pythia-2.8b             # 2.8B - Research model
bigscience/bloom-1b7               # 1.7B - Multilingual
stabilityai/stablelm-2-1_6b        # 1.6B - Recent, efficient
```

### 🔒 **Gated (Needs Approval)**

**If you get approval, these are excellent:**
```
meta-llama/Llama-2-7b-hf           # Requires Meta approval
mistralai/Mistral-7B-v0.1          # Requires Mistral approval
google/gemma-2b                    # Requires Google approval
microsoft/Phi-3-mini-4k-instruct   # May require approval
```

## My Recommended Setup (All Open, No Gatekeeping)

```python
model_ids = [
    # GPT-2 family (baseline)
    "gpt2",                        # 124M
    "gpt2-medium",                 # 355M
    
    # Instruction-tuned
    "google/flan-t5-base",         # 250M
    "google/flan-t5-large",        # 780M
    
    # Research-focused (Pythia)
    "EleutherAI/pythia-410m",      # 410M
    "EleutherAI/pythia-1.4b",      # 1.4B
    
    # State-of-art small
    "microsoft/phi-1_5",           # 1.3B
    "microsoft/phi-2",             # 2.7B
    
    # Open alternatives
    "facebook/opt-350m",           # 350M
    "bigscience/bloomz-560m",      # 560M
]
```

**Total: 10 models, all OPEN, diverse architectures, 0 approvals needed!**

## How to Get Access to Gated Models

### Step 1: Get HuggingFace Token
1. Go to https://huggingface.co/settings/tokens
2. Create new token (read access is enough)
3. Copy the token

### Step 2: Set Environment Variable
```bash
# Add to your ~/.bashrc or ~/.zshrc
export HF_TOKEN="hf_your_token_here"

# Or set temporarily
export HF_TOKEN="hf_your_token_here"
```

### Step 3: Request Access
1. Visit the model page (e.g., https://huggingface.co/meta-llama/Llama-2-7b-hf)
2. Click "Accept terms" or "Request access"
3. Fill out form (for Meta Llama, need to provide use case)
4. **Wait for approval** (can take minutes to days)

### Step 4: Verify Access
```bash
./check_models.sh meta-llama/Llama-2-7b-hf
```

If approved, you'll see:
```
🔓 GATED - You have access
```

## Quick Decision Tree

```
Do you need the absolute best quality?
├─ YES → Apply for Llama-2-7b or Mistral-7B (wait for approval)
└─ NO → Use open models (start testing immediately!)

Do you have lots of time?
├─ YES → Apply for gated models, use while waiting
└─ NO → Stick with open models only

Is this for a thesis/research?
├─ YES → Open models are perfectly valid (widely used in research)
└─ NO → Your choice!
```

## Summary

### ✅ **What Works Now (No Action Needed)**
- All GPT-2 models (gpt2, gpt2-medium, gpt2-large)
- All T5 models (t5-small, flan-t5-base, flan-t5-large)
- EleutherAI Pythia models
- Microsoft Phi models (phi-1, phi-1_5, phi-2)
- Facebook OPT models
- BigScience BLOOM models

### 🔒 **What Needs Approval**
- All Meta Llama models
- Mistral AI models
- Google Gemma models
- Some specialized models

### 💡 **My Recommendation**
**Use the open models** - they are:
- ✅ Immediately available
- ✅ Well-documented
- ✅ Widely used in research
- ✅ Sufficient for thesis comparison
- ✅ No legal/approval delays

**Thesis-ready open setup (10 models):**
```
gpt2, gpt2-medium
google/flan-t5-base, google/flan-t5-large
EleutherAI/pythia-410m, EleutherAI/pythia-1.4b
microsoft/phi-1_5, microsoft/phi-2
facebook/opt-350m, bigscience/bloomz-560m
```

This gives you:
- Size range: 82M → 2.7B parameters
- Architecture variety: GPT-2, T5, Pythia, Phi, OPT, BLOOM
- Task variety: Base models, instruction-tuned, code-focused
- **Zero gatekeeping, start today!**
