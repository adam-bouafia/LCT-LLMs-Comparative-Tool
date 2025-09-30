# LCT Data Directory

This directory contains the organized storage for LLM models and datasets used by the LLM Comparison Tool (LCT).

## Structure

```text
data/
├── models/              # Downloaded LLM models
│   ├── openai-community--gpt2/
│   ├── distilbert--distilgpt2/
│   ├── microsoft--DialoGPT-medium/
│   ├── facebook--blenderbot-400M-distill/
│   └── [other-models]/
├── datasets/            # Downloaded evaluation datasets
│   └── [dataset-directories]/
└── model-selections/    # Saved model selection configurations
    └── [selection-files]/
```

## Model Directory Format

Each model directory follows the HuggingFace cache structure:

- Directory name: `{org}--{model-name}` (slashes replaced with double dashes)
- Contains: model files, tokenizer, configuration
- Metadata: `download_metadata.json` with model type and download info

## Current Models

The following models are currently downloaded and ready for comparison:

| Model ID | Type | Description |
|----------|------|-------------|
| `openai-community/gpt2` | gpt2 | GPT-2 base model |
| `distilbert/distilgpt2` | gpt2 | Distilled GPT-2 |
| `microsoft/DialoGPT-medium` | gpt2 | Conversational GPT-2 |
| `facebook/blenderbot-400M-distill` | blenderbot | Facebook's BlenderBot |

## Management

- **View models**: Use LCT Data Management (option 12) → List Downloaded Models
- **Add models**: Use LCT Data Management (option 12) → Download Model  
- **Remove models**: Use LCT Data Management (option 12) → Cleanup Models
- **Fix metadata**: Run `./fix_model_metadata.py` if models show "Unknown" type

## Storage Space

Current usage: ~7.6 GB for models
Models are shared across experiments to save space.

## Technical Notes

- Models are stored using HuggingFace's standard cache structure
- Each model has a `download_metadata.json` file with type information
- The DataManager class (in `app/src/core/data_manager.py`) handles organization
- Path references use the project root `/data` directory