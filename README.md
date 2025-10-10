# LLM Comparative Tool 🤖⚡

A comprehensive tool for comparing Large Language Models across multiple evaluation metrics including performance, energy consumption, and various NLP benchmarks with research-backed datasets.

## 🌟 Features

- **17 Research-Backed Algorithms**: Dataset-driven LLM comparison with scientific backing
- **15+ Integrated Research Datasets**: HumanEval, GSM8K, HellaSwag, TruthfulQA, SafetyBench, MT-Bench, AlpacaEval, PIQA, CommonsenseQA, CNN/DailyMail, XSum, GLUE, WMT16, MBPP, Winogrande, and more
- **Granular Energy Profiling**: CodeCarbon with hardware-level RAPL counters, GPU tracking, and CO2 emissions
- **Stage-by-Stage Tracking**: Energy monitoring for dataset loading, model initialization, inference, and metrics computation
- **Universal Model Support**: Compatible with all HuggingFace Transformers models
- **Interactive CLI**: User-friendly menu-driven interface with search capabilities and model discovery
- **Comprehensive Results**: Detailed CSV exports with per-algorithm and per-prompt energy data
- **Flexible Configuration**: Save/load experiment configurations for reproducible research
- **Model Discovery**: Search and browse 500,000+ HuggingFace models directly from CLI
- **System Diagnostics**: Built-in compatibility and dependency checking with automated setup

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/adam-bouafia/LCT-LLMs-Comparative-Tool.git
   cd LCT-LLMs-Comparative-Tool
   ```

2. **Run the tool**
   ```bash
   ./start_lct.sh
   ```

   The launcher will automatically:
   - Create a virtual environment (if needed)
   - Install dependencies
   - Launch the interactive menu

## 📊 Research-Backed Evaluation Algorithms

All algorithms are backed by established research datasets for scientific validity. Each algorithm is paired with specific datasets to ensure accurate and reproducible evaluation.

### Text Evaluation Metrics (5 algorithms)

1. **BLEU Score** - Translation quality measurement
   - Dataset: **WMT16** (de-en, ro-en translation pairs)
   - Reference-based n-gram overlap metric

2. **ROUGE Score** - Summarization evaluation
   - Datasets: **CNN/DailyMail** (300K+ articles), **XSum** (226K BBC articles)
   - Measures recall-oriented overlap for abstractive/extractive summarization

3. **BERTScore** - Semantic similarity using contextual embeddings
   - Dataset: **GLUE MRPC & STS-B** (Microsoft Research Paraphrase Corpus, Semantic Textual Similarity Benchmark)
   - Context-aware similarity using pre-trained BERT models

4. **Semantic Similarity** - Cosine similarity with sentence transformers
   - Dataset: **GLUE STS-B** (5,749 sentence pairs with human similarity scores)
   - Uses sentence-transformers/all-MiniLM-L6-v2 embeddings

5. **STS Algorithm** - Semantic Textual Similarity evaluation
   - Dataset: **GLUE STS-B** (Semantic Textual Similarity Benchmark)
   - Correlation with human similarity judgments

### LLM-Based Evaluation (2 algorithms)

6. **Pairwise Comparison** - Head-to-head model evaluation
   - Dataset: **AlpacaEval** (805 instruction-following pairs)
   - GPT-4 as judge for comparing model outputs

7. **LLM-as-Judge** - AI-powered quality assessment
   - Dataset: **MT-Bench** (3,355 human judgments, 80 multi-turn conversations)
   - Multi-turn conversation quality evaluation

### Task-Specific Benchmarks (10 algorithms)

8. **Code Generation** - Programming problem solving
   - Dataset: **HumanEval** (164 hand-written Python programming problems)
   - Pass@k metric with unit test validation

9. **Mathematical Reasoning** - Math problem solving capability
   - Dataset: **GSM8K** (8.5K grade school math word problems)
   - Chain-of-thought reasoning evaluation

10. **Commonsense Reasoning** - Physical and social common sense
    - Dataset: **HellaSwag** (70K+ commonsense inference scenarios)
    - Context completion with adversarial filtering

11. **Safety Alignment** - Safety and ethics assessment
    - Dataset: **SafetyBench** (11,435 safety scenarios in English & Chinese)
    - Multi-category safety evaluation (7 categories, 46 tasks)

12. **Truthfulness** - Factual accuracy and hallucination detection
    - Dataset: **TruthfulQA** (817 questions spanning 38 categories)
    - Tests resistance to generating false information

13. **Physical Commonsense** - Physical reasoning capability
    - Dataset: **PIQA** (Physical Interaction QA - 16K questions)
    - Physical world understanding and causal reasoning

14. **General Commonsense** - Question answering with world knowledge
    - Dataset: **CommonsenseQA** (12K questions with 5 answer choices)
    - Requires complex reasoning over commonsense knowledge

15. **Code Reasoning** - Multi-step code problem solving
    - Dataset: **MBPP** (Mostly Basic Python Problems - 974 problems)
    - Entry-level programming tasks with test-based validation

16. **Sentence Completion** - Commonsense sentence completion
    - Dataset: **Winogrande** (44K+ problems, XL variant)
    - Winograd schema challenge for pronoun resolution

17. **Robustness Testing** - Adversarial and edge case evaluation
    - Datasets: **HH-RLHF** (Anthropic Helpful & Harmless - 170K conversations)
    - Tests model safety, helpfulness, and robustness to adversarial inputs

## 🔋 Energy Profiling

**CodeCarbon** - The sole energy profiler for comprehensive hardware-level measurement:

### Key Capabilities

✓ **RAPL (Running Average Power Limit)**: Direct CPU energy from Intel/AMD hardware counters  
✓ **GPU Tracking**: NVIDIA GPU energy via nvidia-smi/NVML API  
✓ **CO2 Emissions**: Automatic carbon footprint calculation with regional grid data  
✓ **No Root Required**: Reads `/sys/class/powercap/intel-rapl/*/energy_uj` files directly  
✓ **Research-Validated**: Accuracy verified against physical power meters (2025 studies)

### Detailed Features

- **Hardware-level precision**: Measurements in microjoules (μJ), updated every ~1ms
- **Comprehensive coverage**: CPU package, cores, DRAM, integrated GPU, and discrete GPU
- **NVIDIA GPU support**: Direct NVML interface for real-time GPU power consumption
- **Memory monitoring**: RAM usage tracking throughout experiment lifecycle
- **Carbon intelligence**: Automatic CO2 calculations with regional electricity grid emissions
- **Real-time tracking**: Live energy monitoring during experiments with progress updates
- **Granular breakdowns**: Per-model, per-algorithm, and per-prompt energy analysis
- **Stage-by-stage profiling**: Separate tracking for dataset loading, model initialization, inference, and metrics computation
- **Export-ready data**: Detailed CSV exports with energy metrics for research publication

## 📁 Project Structure

```
LLM-Comparative-Tool/
├── 🚀 start_lct.sh                    # Main launcher script
├── � run_lct_cli.sh                  # Command-line interface launcher
├── �📋 README.md                       # Project documentation
├── ⚖️  LICENSE                         # License file
├── 📄 CITATION.cff                    # Citation information
├── � pyrightconfig.json              # Python type checking configuration
├── �📱 app/                            # Application code
│   └── src/                           # Source code
│       ├── ui/                        # User interface components
│       │   └── interactive_lct.py     # Main interactive tool
│       ├── llm_runner/               # LLM comparison engine
│       │   ├── algorithms/           # Evaluation algorithms
│       │   ├── cli/                  # Command-line interface
│       │   ├── data/                 # Data management
│       │   └── loaders/              # Model loaders
│       └── experiment_runner/        # Experiment orchestration
├── 📜 scripts/                        # Utility scripts
│   ├── analyze_algorithm_datasets.py # Algorithm analysis
│   ├── dataset_manager.py            # Dataset management
│   ├── check_lct_status.py          # System status checker
│   └── validate_imports.py          # Import validation
├── ⚙️  config/                        # Configuration files
├── 📊 data/                           # Data and datasets (gitignored)
│   ├── huggingface/                  # HuggingFace cache
│   ├── research_datasets/            # Research datasets
│   └── README.md                     # Data directory info
├── 🧪 experiments/                    # Experiment results
└── 🔋 .venv/                         # Virtual environment (gitignored)
```


## 🛠️ Usage

1. **Launch the tool**: `./start_lct.sh`
2. **Set experiment name** and select models to compare
3. **Choose evaluation algorithms** from research-backed options
4. **Configure energy profiling** (optional)
5. **Save/load configurations** for reproducible experiments
6. **Run experiment** and view real-time progress
7. **Explore results** with built-in analysis tools

### Alternative Usage

You can also run components directly:

```bash
# Activate virtual environment
source .venv/bin/activate

# Run specific components
python3 app/src/ui/interactive_lct.py           # Interactive menu
python3 app/src/tools/run_lct.py               # Direct runner
python3 app/src/ui/results_explorer.py         # Results analysis
```


## � Screenshots

### Main Interface
![CLI Menu](https://i.imgur.com/a2gxSqe.png)
*Interactive CLI menu with all available options*

### Experiment Setup

#### Set Experiment Name
![Set Experiment Name](https://i.imgur.com/xgWybR7.png)
*Configure your experiment with a descriptive name*

#### Select Models
![Select LLMs Models](https://i.imgur.com/4N67y1H.png)
*Choose supported LLM models From HF*

#### Search HuggingFace Models
![Search Models on HuggingFace](https://i.imgur.com/VdKtwTy.png)
*Discover and search models directly from HuggingFace*

### Configuration Options

#### Select Prompts
![Select Prompts](https://i.imgur.com/BiEbKRv.png)
*Choose evaluation prompts for your experiment*

#### Choose Algorithms
![Select Algorithms](https://i.imgur.com/rw2THSm.png)
*Select from 12 research-backed evaluation algorithms*

#### Configure Parameters
![Configure Parameters](https://i.imgur.com/0waa7jW.png)
*Fine-tune experiment parameters and settings*

### Advanced Features

#### Energy Profiling
![Energy Profiling](https://i.imgur.com/Ix6ZaGc.png)
*Configure energy consumption monitoring with multiple profilers*

#### Save Configuration
![Save Config](https://i.imgur.com/QAhQP4Y.png)
*Save experiment configurations for reproducible research*

#### Load Configuration
![Load Config](https://i.imgur.com/ryOLeuw.png)
*Load previously saved experimental setups*

### Tools & Setup

#### Install Dependencies
![Install Tools](https://i.imgur.com/0PeA9j6.png)
*Automated installation of required tools and dependencies*

#### System Diagnostics
![System Diagnostics](http://i.imgur.com/XFLhsq2.png)
*Comprehensive system compatibility and performance checks*

#### HuggingFace Authentication
![HF Auth](https://i.imgur.com/LPirbhs.png)
*Set up HuggingFace authentication for model access*

### Results & Information

#### Results Explorer
![Results Explorer](https://i.imgur.com/F10hgal.png)
*Interactive analysis and visualization of experiment results*

#### About Tool
![About Tool](https://i.imgur.com/IDhIXr0.png)
*Developer information and tool details*


## 📈 Results

The tool generates comprehensive, publication-ready results including:

- **Complete Algorithm Coverage**: Individual run data with all 17 research-backed algorithms
- **Statistical Analysis**: Aggregated statistics, model comparisons, and performance rankings
- **Energy Profiling**: Detailed energy consumption profiles (CPU, GPU, DRAM, memory, carbon footprint)
- **Per-Stage Metrics**: Energy tracking for dataset loading, model initialization, inference, and metrics computation
- **Dataset-Specific Results**: Performance metrics tied to specific research datasets (15+ datasets)
- **CSV Exports**: Detailed exports with algorithm-specific metrics, timestamps, and energy data
- **Real-Time Monitoring**: Live progress tracking during experiments with stage completion indicators
- **Interactive Explorer**: Built-in results analysis tool for post-experiment visualization
- **Reproducibility**: All configurations and parameters saved with results for research reproducibility

## 🔧 Configuration

- **Experiments**: Configure in `experiments/` directory
- **Models**: Preset configurations in `data/model-selections/`
- **Dependencies**: Listed in `config/requirements.txt`
- **Setup**: Python packaging in `config/setup.py`

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## �‍💻 Developer Information

**Developed by Adam Bouafia**

🔗 **Connect with me:**
- 🌐 **Developer Portfolio**: [https://adam-bouafia.github.io/](https://adam-bouafia.github.io/)
- 💼 **LinkedIn Profile**: [https://www.linkedin.com/in/adam-bouafia-b597ab86/](https://www.linkedin.com/in/adam-bouafia-b597ab86/)
- 🧪 **Experiment Runner Framework**: [https://github.com/S2-group/experiment-runner](https://github.com/S2-group/experiment-runner)

💝 **Support Development:**
- 💳 **Donate via PayPal**: [https://paypal.me/AdamBouafia](https://paypal.me/AdamBouafia)
- ⭐ **Star this repository** if you find it useful!
- 🗣️ **Share feedback** to help improve the tool

*Your donations and support help maintain and improve this tool for the research community!*

## �📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this tool in your research, please cite:

```bibtex
@software{llm_comparative_tool_2025,
  title={LLM Comparative Tool: Research-Backed Large Language Model Evaluation},
  author={Adam Bouafia},
  year={2025},
  url={https://github.com/adam-bouafia/LCT-LLMs-Comparative-Tool},
  note={Comprehensive LLM evaluation with 17 research-backed algorithms and 15+ integrated datasets including HumanEval, GSM8K, HellaSwag, TruthfulQA, SafetyBench, MT-Bench, AlpacaEval, PIQA, CommonsenseQA, CNN/DailyMail, XSum, GLUE, WMT16, MBPP, Winogrande, and HH-RLHF. Features hardware-level energy profiling with CodeCarbon.}
}
```

## 🔗 Quick Links

- [Configuration Files](config/)
- [Example Experiments](experiments/)
- [Model Compatibility](data/model-selections/)

---

**Built with ❤️**
