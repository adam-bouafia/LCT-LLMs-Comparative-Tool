# LLM Comparative Tool 🤖⚡

A comprehensive tool for comparing Large Language Models across multiple evaluation metrics including performance, energy consumption, and various NLP benchmarks with research-backed datasets.

## 🌟 Features

- **17 Research-Backed Algorithms**: Dataset-driven LLM comparison with scientific backing
- **8 Integrated Research Datasets**: HumanEval, GSM8K, HellaSwag, TruthfulQA, SafetyBench, MT-Bench, AlpacaEval, Arena Preferences
- **Granular Energy Profiling**: Stage-by-stage energy tracking (dataset loading, model init, inference, metrics)
- **Universal Model Support**: Compatible with all HuggingFace Transformers models
- **Interactive CLI**: User-friendly menu-driven interface with search capabilities
- **Comprehensive Results**: Detailed CSV exports with per-algorithm and per-prompt energy data
- **Flexible Configuration**: Save/load experiment configurations
- **Model Discovery**: Search and browse HuggingFace models directly from CLI
- **System Diagnostics**: Built-in compatibility and dependency checking

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

All algorithms are backed by established research datasets for scientific validity:

### Text Evaluation Metrics (5)
- **BLEU Score**: Translation quality with reference data
- **ROUGE Score**: Summarization evaluation (CNN/DailyMail, XSum)
- **BERTScore**: Semantic similarity with STS Benchmark
- **Semantic Similarity**: Contextual understanding (STS Benchmark)
- **STS Algorithm**: Semantic Textual Similarity evaluation

### LLM-Based Evaluation (2)
- **Pairwise Comparison**: Head-to-head evaluation (AlpacaEval - 805 pairs)
- **LLM-as-Judge**: AI-powered evaluation (MT-Bench - 3,355 judgments)

### Task-Specific Benchmarks (5)
- **Code Generation**: Programming capability (HumanEval - 164 problems)
- **Mathematical Reasoning**: Math problem solving (GSM8K - 1,319 problems)
- **Commonsense Reasoning**: Common sense understanding (HellaSwag - 10,042 scenarios)
- **Safety Alignment**: Safety and ethics assessment (SafetyBench - 11,435 scenarios)
- **Truthfulness**: Factual accuracy assessment (TruthfulQA - 817 questions)

## 🔋 Energy Profiling

**CodeCarbon** - Comprehensive hardware-level energy measurement:

✓ **RAPL (Running Average Power Limit)**: Real CPU energy from hardware counters  
✓ **GPU Tracking**: NVIDIA GPU energy via nvidia-smi  
✓ **CO2 Emissions**: Automatic carbon footprint calculation  
✓ **No Root Required**: Reads `/sys/class/powercap` files directly  
✓ **Research-Validated**: Compared against physical power meters (2025 studies)

Features:
- Hardware-level measurements in microjoules (μJ)
- CPU package, core, DRAM, and integrated GPU energy
- NVIDIA discrete GPU tracking via NVML
- Memory consumption monitoring
- Carbon footprint with regional grid data
- Real-time energy monitoring during experiments
- Detailed energy breakdowns per model and algorithm
- Stage-by-stage tracking (dataset loading, model init, inference, metrics)

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

The tool generates comprehensive results including:

- Individual run data with all 12 research-backed algorithms
- Aggregated statistics and model comparisons
- Energy consumption profiles (CPU, memory, carbon footprint)
- Detailed CSV exports with algorithm-specific metrics
- Real-time progress tracking during experiments
- Built-in results explorer for interactive analysis

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
  note={Comprehensive LLM evaluation with 12 research-backed algorithms and 7 integrated datasets}
}
```

## 🔗 Quick Links

- [Configuration Files](config/)
- [Example Experiments](experiments/)
- [Model Compatibility](data/model-selections/)

---

**Built with ❤️**
