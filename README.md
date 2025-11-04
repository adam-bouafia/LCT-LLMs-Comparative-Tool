# LLM Comparative Tool 🤖⚡

A comprehensive tool for comparing Large Language Models across multiple evaluation metrics including performance, energy consumption, and various NLP benchmarks with research-backed datasets.

## 🌟 Features

- **17 Research-Backed Algorithms**: Dataset-driven LLM comparison with scientific backing
- **15+ Integrated Research Datasets**: HumanEval, GSM8K, HellaSwag, TruthfulQA, SafetyBench, MT-Bench, AlpacaEval, PIQA, CommonsenseQA, CNN/DailyMail, XSum, GLUE, WMT16, MBPP, Winogrande, and more
- **Comprehensive Environmental Tracking**: Full lifecycle environmental impact analysis with water, PUE, and carbon metrics
- **35+ Global Regions**: Production-ready regional environmental multipliers covering all major data center locations worldwide
- **Multi-Select Energy Configuration**: Independent toggles for CodeCarbon and Environmental Tracking with intuitive UI
- **Granular Energy Profiling**: CodeCarbon with hardware-level RAPL counters, GPU tracking, and CO2 emissions
- **Stage-by-Stage Tracking**: Energy monitoring for dataset loading, model initialization, inference, and metrics computation
- **Universal Model Support**: Compatible with all HuggingFace Transformers models
- **Interactive CLI**: User-friendly menu-driven interface with search capabilities and model discovery
- **Enhanced Status Display**: Real-time configuration status showing experiment name, selected components, and environmental settings
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

1. **Pairwise Comparison** - Head-to-head model evaluation
   - Dataset: **AlpacaEval** (805 instruction-following pairs)
   - GPT-4 as judge for comparing model outputs

2. **LLM-as-Judge** - AI-powered quality assessment
   - Dataset: **MT-Bench** (3,355 human judgments, 80 multi-turn conversations)
   - Multi-turn conversation quality evaluation

### Task-Specific Benchmarks (10 algorithms)

1. **Code Generation** - Programming problem solving
   - Dataset: **HumanEval** (164 hand-written Python programming problems)
   - Pass@k metric with unit test validation

2. **Mathematical Reasoning** - Math problem solving capability
   - Dataset: **GSM8K** (8.5K grade school math word problems)
   - Chain-of-thought reasoning evaluation

3. **Commonsense Reasoning** - Physical and social common sense
    - Dataset: **HellaSwag** (70K+ commonsense inference scenarios)
    - Context completion with adversarial filtering

4. **Safety Alignment** - Safety and ethics assessment
    - Dataset: **SafetyBench** (11,435 safety scenarios in English & Chinese)
    - Multi-category safety evaluation (7 categories, 46 tasks)

5. **Truthfulness** - Factual accuracy and hallucination detection
    - Dataset: **TruthfulQA** (817 questions spanning 38 categories)
    - Tests resistance to generating false information

6. **Physical Commonsense** - Physical reasoning capability
    - Dataset: **PIQA** (Physical Interaction QA - 16K questions)
    - Physical world understanding and causal reasoning

7. **General Commonsense** - Question answering with world knowledge
    - Dataset: **CommonsenseQA** (12K questions with 5 answer choices)
    - Requires complex reasoning over commonsense knowledge

8. **Code Reasoning** - Multi-step code problem solving
    - Dataset: **MBPP** (Mostly Basic Python Problems - 974 problems)
    - Entry-level programming tasks with test-based validation

9. **Sentence Completion** - Commonsense sentence completion
    - Dataset: **Winogrande** (44K+ problems, XL variant)
    - Winograd schema challenge for pronoun resolution

10. **Robustness Testing** - Adversarial and edge case evaluation
    - Datasets: **HH-RLHF** (Anthropic Helpful & Harmless - 170K conversations)
    - Tests model safety, helpfulness, and robustness to adversarial inputs

## 🐳 Docker Deployment

LCT is available as a Docker image for easy deployment without cloning the repository.

### Quick Start with Docker

**Option 1: Pull and Run (Recommended for Users)**

```bash
# Pull the latest image from Docker Hub
docker pull adambouafia/lct:latest

# Run the tool
docker run -it --rm adambouafia/lct:latest
```

**Option 2: Build from Source (For Development)**

```bash
# Clone the repository
git clone https://github.com/adam-bouafia/LCT-LLMs-Comparative-Tool.git
cd LCT-LLMs-Comparative-Tool

# Build and run with Docker Compose
docker-compose up

# Or use the helper script
./docker-build.sh
./docker-run.sh
```

### Docker Features

✓ **Multi-stage build**: Optimized image size with separate build and runtime stages  
✓ **Non-root user**: Security-first approach with dedicated `lct` user (uid 1000)  
✓ **GPU support**: Optional NVIDIA GPU access for accelerated inference  
✓ **Energy profiling**: Privileged mode for RAPL hardware counter access  
✓ **Persistent volumes**: Automatic caching of models, datasets, and results  
✓ **Interactive mode**: Full CLI functionality in containerized environment  
✓ **Health checks**: Built-in container health monitoring  

### Installation Options

#### Option 1: Docker Compose (Recommended)

```bash
# Start the container
docker-compose up

# Run in detached mode
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the container
docker-compose down
```

#### Option 2: Direct Docker Run

```bash
# Build the image
docker build -t lct:latest .

# Run with basic settings
docker run -it --rm lct:latest

# Run with volume mounts for persistence
docker run -it --rm \
  -v $(pwd)/experiments:/app/experiments \
  -v $(pwd)/saved_configs:/app/saved_configs \
  lct:latest
```

#### Option 3: Helper Scripts

```bash
# Build the image
./docker-build.sh

# Run with default settings
./docker-run.sh

# Run with GPU support
./docker-run.sh --gpu

# Run with energy profiling (privileged mode)
./docker-run.sh --privileged

# Open a shell in the container
./docker-shell.sh
```

### GPU Support

To enable GPU support for accelerated inference:

**Prerequisites:**
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit installed on host

**Enable in docker-compose.yml:**
```yaml
services:
  lct:
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
```

**Or run with script:**
```bash
./docker-run.sh --gpu
```

### Volume Mounts

The Docker setup uses several volumes for data persistence:

| Volume | Purpose | Type |
|--------|---------|------|
| `huggingface_cache` | HuggingFace models & datasets | Named volume |
| `models_cache` | PyTorch models | Named volume |
| `./experiments` | Experiment results | Bind mount |
| `./saved_configs` | Configuration files | Bind mount |
| `./logs` | Application logs | Bind mount |

**Benefits:**
- Models downloaded once, persist across container restarts
- Experiment results saved to host filesystem
- Easy access to configuration files from host

### Environment Variables

Configure LCT behavior with environment variables:

```bash
# HuggingFace configuration
HF_HOME=/data/huggingface                    # Cache directory
HF_TOKEN=hf_your_token_here                  # Optional: for private models
TRANSFORMERS_CACHE=/data/huggingface/transformers
HF_DATASETS_CACHE=/data/huggingface/datasets

# PyTorch configuration
TORCH_HOME=/data/models

# Python configuration
PYTHONUNBUFFERED=1                           # Real-time output
```

Create a `.env` file in the project root for custom configuration:

```bash
# .env
HF_TOKEN=hf_your_token_here
CUDA_VISIBLE_DEVICES=0
```

### Energy Profiling in Docker

Energy profiling requires access to hardware counters:

**Option 1: Privileged Mode (Full Access)**
```bash
docker run -it --rm --privileged lct:latest
```

**Option 2: Selective Capabilities (More Secure)**
```bash
docker run -it --rm \
  --cap-add=SYS_ADMIN \
  --device=/dev/cpu/0/msr:/dev/cpu/0/msr \
  lct:latest
```

**Note:** Energy profiling may have limited functionality in containerized environments. For full RAPL access, consider running on bare metal or with privileged mode.

### Troubleshooting

**Container won't start:**
```bash
# Check Docker service
sudo systemctl status docker

# View container logs
docker logs lct-tool

# Check image build
docker images | grep lct
```

**Permission issues:**
```bash
# Ensure scripts are executable
chmod +x docker-*.sh

# Check volume permissions
ls -la experiments/ saved_configs/
```

**GPU not detected:**
```bash
# Verify NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Check NVIDIA Container Toolkit
nvidia-ctk --version
```

**Out of disk space:**
```bash
# Clean up Docker resources
docker system prune -a

# Remove unused volumes
docker volume prune

# Check volume sizes
docker system df
```

## �🔋 Energy Profiling & Environmental Impact

**CodeCarbon + Environmental Tracking** - Production-ready comprehensive measurement with global environmental impact analysis:

### Key Capabilities

✓ **RAPL (Running Average Power Limit)**: Direct CPU energy from Intel/AMD hardware counters  
✓ **GPU Tracking**: NVIDIA GPU energy via nvidia-smi/NVML API  
✓ **CO2 Emissions**: Automatic carbon footprint calculation with regional grid data  
✓ **Water Usage Tracking (WUE)**: On-site cooling + off-site electricity generation water consumption  
✓ **Power Usage Effectiveness (PUE)**: Real data center infrastructure overhead (8-27% overhead)  
✓ **Eco-Efficiency Scoring**: Performance per unit of environmental cost (DEA methodology)  
✓ **35+ Global Regions**: Comprehensive worldwide coverage with accurate regional multipliers  
✓ **Multi-Select Configuration**: Independent toggles for CodeCarbon and Environmental Tracking  
✓ **Real-Time Status Display**: Live environmental settings shown in main menu  
✓ **Scaled Impact Projections**: Estimate production-scale environmental footprint  
✓ **No Root Required**: Reads `/sys/class/powercap/intel-rapl/*/energy_uj` files directly  
✓ **Production Ready**: All demo files removed, ready for enterprise deployment

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
- **Water footprint**: Track liters of water consumed (cooling + electricity generation)
- **Infrastructure overhead**: Account for real data center PUE (1.08-1.27x multiplier)
- **Eco-efficiency ranking**: Compare models by performance per environmental cost
- **Production projections**: Estimate annual impact if scaled to millions of queries/day

## 🌍 Environmental Impact Analysis

Comprehensive environmental metrics integrated into LCT:

### Water Usage Tracking
- **On-site water (WUE site)**: Cooling systems in data centers (0.05-1.20 L/kWh)
- **Off-site water (WUE source)**: Electricity generation (0.5-6.0 L/kWh)
- **Total footprint**: Per-experiment and scaled production estimates

### Power Usage Effectiveness (PUE)
- **Infrastructure overhead**: Cooling, lighting, power distribution (8-27% overhead)
- **Global regional variation**: 50+ regions from Iceland (1.08) to India (1.27)
- **Realistic estimates**: True data center energy vs. IT equipment only

### Eco-Efficiency Scoring
- **Performance vs. cost**: Ranks models by accuracy/throughput per unit of water/energy/carbon
- **Multi-dimensional**: Balances multiple environmental metrics simultaneously
- **DEA methodology**: Data Envelopment Analysis for comprehensive efficiency scoring

### Regional Coverage (35 Global Regions)

#### North America (6 regions)
- **USA East Coast** (CIF: 0.340) - Virginia, N. Carolina data centers
- **USA West Coast** (CIF: 0.289) - California, Oregon renewable-heavy
- **USA Central** (CIF: 0.415) - Texas, Midwest coal/gas mix
- **Canada East** (CIF: 0.065) - Quebec hydroelectric dominance
- **Canada West** (CIF: 0.080) - BC clean energy grid
- **Mexico** (CIF: 0.450) - Mixed fossil/renewable

#### Europe - West (7 regions)
- **Europe West** (CIF: 0.295) - Netherlands, Belgium mixed
- **United Kingdom** (CIF: 0.233) - Wind + nuclear transition
- **Ireland** (CIF: 0.338) - Growing wind capacity
- **France** (CIF: 0.055) - Nuclear-dominated grid (cleanest in EU)
- **Germany** (CIF: 0.338) - Renewable transition ongoing
- **Netherlands** (CIF: 0.305) - Natural gas heavy
- **Switzerland** (CIF: 0.020) - Hydro + nuclear (ultra-clean)

#### Europe - North (3 regions)
- **Europe North** (CIF: 0.125) - Scandinavia average
- **Nordic Countries** (CIF: 0.035) - Norway, Sweden, Finland hydro
- **Iceland** (CIF: 0.010) - 100% geothermal/hydro (cleanest globally)

#### Europe - East & South (2 regions)
- **Europe East** (CIF: 0.455) - Poland coal-heavy
- **Europe South** (CIF: 0.345) - Italy, Spain, Greece mixed

#### Asia Pacific - China (3 regions)
- **China North** (CIF: 0.650) - Coal-dominant Beijing region
- **China East** (CIF: 0.555) - Shanghai industrial corridor
- **China South** (CIF: 0.582) - Guangdong manufacturing hub

#### Asia Pacific - Other (5 regions)
- **Japan** (CIF: 0.468) - Post-Fukushima fossil reliance
- **South Korea** (CIF: 0.428) - Industrial powerhouse
- **Singapore** (CIF: 0.392) - Tropical data center hub
- **Australia East** (CIF: 0.720) - NSW coal-heavy
- **Australia Southeast** (CIF: 0.685) - Victoria brown coal

#### India (3 regions)
- **India West** (CIF: 0.675) - Maharashtra Mumbai region
- **India South** (CIF: 0.710) - Tamil Nadu, Bangalore tech hub
- **India Central** (CIF: 0.735) - Highest coal dependency

#### Middle East & Africa (3 regions)
- **United Arab Emirates** (CIF: 0.415) - Dubai data centers
- **Saudi Arabia** (CIF: 0.455) - Oil-powered infrastructure
- **South Africa** (CIF: 0.950) - Highest carbon intensity globally (coal grid)

#### South America (2 regions)
- **Brazil South** (CIF: 0.075) - Hydroelectric paradise
- **Brazil Southeast** (CIF: 0.085) - São Paulo clean grid

#### Custom (1 option)
- **Custom Region** - Define your own multipliers for specific data centers

**PUE Range**: 1.08 (Iceland) to 1.28 (India Central)  
**WUE Range**: 0.55 L/kWh (Iceland) to 7.45 L/kWh (India Central)  
**CIF Range**: 0.010 kgCO2e/kWh (Iceland) to 0.950 kgCO2e/kWh (South Africa)

### Production Scale Projections
Example: 1M queries/day for 1 year
- **Energy**: 3,650 MWh (equivalent to 1,200 US homes)
- **Water**: 15 million liters (41,000 people's annual drinking water)
- **Carbon**: 1,287 metric tons CO2e (requires 46,900 trees to offset)

**Learn more**: See [Environmental Impact Guide](docs/ENVIRONMENTAL_IMPACT_GUIDE.md)

## 🆕 Recent Updates (November 2025)

### Enhanced Environmental Tracking
- ✅ **35 Global Regions** - Expanded from 5 to 35 regions covering all major data center locations worldwide
- ✅ **Production-Ready** - Removed all demo files and research paper references
- ✅ **Multi-Select UI** - Independent toggles for CodeCarbon and Environmental Tracking
- ✅ **Enhanced Status Display** - Main menu now shows experiment name and full environmental configuration
- ✅ **No Hanging** - Fixed subprocess handling for smooth experiment creation (no manual Enter needed)

### UI/UX Improvements
- **Real-time Status**: Experiment name displayed in status column `✅ (experiment_name)`
- **Environmental Status**: Shows active features `✅ (codecarbon+env: France)`
- **Smart Validation**: Environmental tracking requires CodeCarbon to be enabled first
- **Regional Selection**: Organized by geography with 10 region groups for easy navigation
- **Live Configuration**: Current settings displayed at top of configuration menu

### Technical Enhancements
- **Subprocess Fix**: Added `stdin=subprocess.DEVNULL` to prevent hanging on CLI prompts
- **Status Helper**: Centralized `_get_energy_status()` method for consistent display
- **Config Persistence**: Environmental settings saved/loaded with experiment configurations
- **Region Validation**: All 35 regions tested and validated with accurate multipliers

## 📁 Project Structure

```text
├── 📋 README.md                       # Project documentation
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
2. **Set experiment name** - Now displays in status: `✅ (your_experiment_name)`
3. **Select models to compare** - Shows count: `✅ (3 selected)`
4. **Choose evaluation algorithms** from research-backed options
5. **Configure energy profiling** with multi-select:
   - Toggle CodeCarbon energy tracking independently
   - Toggle Environmental Tracking (water, PUE, eco-efficiency) independently
   - Select from 35 global regions for accurate environmental multipliers
   - Status shows: `✅ (codecarbon+env: France)` when both enabled
6. **Save/load configurations** for reproducible experiments
7. **Run experiment** without hanging (automatic subprocess handling)
8. **Explore results** with built-in analysis tools

### Multi-Select Energy Configuration

The new energy profiling menu allows independent control:

```
╭────────┬───────────────────────────────┬─────────────────────────────────╮
│ Option │ Action                        │ Description                     │
├────────┼───────────────────────────────┼─────────────────────────────────┤
│ 1      │ Toggle Energy Profiler        │ Enable/disable CodeCarbon       │
│ 2      │ Toggle Environmental Tracking │ Enable/disable water/PUE/carbon │
│ 3      │ Select Region                 │ Choose from 35 global regions   │
│ 0      │ Save & Return                 │ Apply settings                  │
╰────────┴───────────────────────────────┴─────────────────────────────────╯
```

**Configuration States:**
- **CodeCarbon OFF, Environmental OFF**: No tracking
- **CodeCarbon ON, Environmental OFF**: Basic energy + CO2 only
- **CodeCarbon ON, Environmental ON**: Full environmental impact with water, PUE, regional carbon, and eco-efficiency

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

## 💻 Developer Information

### Developed by Adam Bouafia

🔗 **Connect with me:**

- 🌐 **Developer Portfolio**: [https://adam-bouafia.github.io/](https://adam-bouafia.github.io/)
- 💼 **LinkedIn Profile**: [https://www.linkedin.com/in/adam-bouafia-b597ab86/](https://www.linkedin.com/in/adam-bouafia-b597ab86/)
- 🧪 **Experiment Runner Framework**: [https://github.com/S2-group/experiment-runner](https://github.com/S2-group/experiment-runner)

💝 **Support Development:**

- 💳 **Donate via PayPal**: [https://paypal.me/AdamBouafia](https://paypal.me/AdamBouafia)
- ⭐ **Star this repository** if you find it useful!
- 🗣️ **Share feedback** to help improve the tool

*Your donations and support help maintain and improve this tool for the research community!*

## 📄 License

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

## Built with ❤️
