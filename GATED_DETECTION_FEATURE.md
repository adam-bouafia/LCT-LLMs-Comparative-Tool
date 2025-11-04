# Gated Model Detection - Feature Update

## What's New

The interactive LCT tool now **automatically detects** if HuggingFace models are gated and shows their access status in real-time!

## Features

### 🔍 In Model Search

When searching for models, you'll now see an **"Access" column** with these statuses:

- **🔓 Open** - No restrictions, ready to use immediately
- **🔒 Gated** - Requires approval (you need to visit the model page)
- **🔐 Gated (Access)** - Gated but you have access via your HF_TOKEN
- **❓ Unknown** - Status couldn't be determined

### ⚠️ Selection Warning

When you try to add a gated model without access, you'll get a warning:

```
⚠️  WARNING: meta-llama/Llama-2-7b-hf is GATED
   Status: 🔒 Gated
   You need to request access at: https://huggingface.co/meta-llama/Llama-2-7b-hf
   The model will fail to load without approval!
   
   Add anyway? [y/n]:
```

This prevents accidentally adding models that will fail during experiments!

## How It Works

1. **Uses HuggingFace API** - Leverages your HF_TOKEN if set
2. **Real-time checking** - Checks each model as displayed
3. **Saves metadata** - Stores gated status with model info

## Setting Up HF_TOKEN

To enable full gated model detection:

```bash
# Get token from: https://huggingface.co/settings/tokens
export HF_TOKEN="hf_your_token_here"

# Or set permanently in ~/.bashrc
echo 'export HF_TOKEN="hf_your_token_here"' >> ~/.bashrc
```

## Benefits

✅ **No surprises** - Know if a model requires approval before adding it
✅ **Smart warnings** - Get alerted when selecting gated models
✅ **Better planning** - Choose open models for immediate testing
✅ **Token awareness** - See which gated models you already have access to

## Example Use Case

**Before:**
- Add meta-llama/Llama-2-7b-hf to experiment
- Run experiment
- ❌ Error: "Access to model restricted"
- Frustration!

**After:**
- Search for "llama"
- See "🔒 Gated" status
- Get warning with link to request access
- Choose to either:
  - Request access and wait
  - Use open alternative like "TinyLlama" (shows "🔓 Open")

## Related Files

- `app/src/ui/interactive_lct.py` - Main implementation
- `app/src/tools/gated_model_checker.py` - Standalone checker
- `GATED_MODELS_SOLUTION.md` - Comprehensive guide
- `recommended_open_models.txt` - Curated open model list

## Testing

```bash
# Start interactive tool
./start_lct.sh

# In the tool:
# 1. Go to "Select Models"
# 2. Press 's' to search
# 3. Search for "llama"
# 4. See the "Access" column showing gated status
# 5. Try to add a gated model - you'll get a warning!
```

Enjoy hassle-free model selection! 🎉
