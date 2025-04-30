# Genesis OS

Genesis OS is an advanced artificial intelligence system that combines multiple AI components to create a comprehensive learning and decision-making platform. It integrates vision perception, memory systems, decision engines, and evolutionary algorithms to create a robust and adaptable AI system.

## Features

- **Multi-Modal Learning**: Combines vision, text, and audio processing capabilities
- **Advanced Memory System**: Implements both short-term and long-term memory with neural Turing machines
- **Decision Engine**: Utilizes multiple reinforcement learning algorithms (DQN, PPO, SAC)
- **Evolutionary Learning**: Implements EVO-Forge for continuous improvement
- **Resource Management**: Efficient monitoring and optimization of system resources
- **Task Management**: Asynchronous task execution and scheduling
- **Learning Management**: Comprehensive tracking of learning metrics and progress

## Installation

### Quick Installation

```bash
pip install genius-ai
```

### Development Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/genesis-os.git
cd genesis-os
```

2. Create a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install development dependencies:
```bash
pip install -e ".[dev]"
```

### GPU Support (Recommended)

For GPU support, install PyTorch with CUDA:
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### System Requirements

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- 8GB RAM minimum (16GB recommended)
- 20GB free disk space

## Quick Start

1. Basic usage:
```python
from genius_ai.core.genesis import GenesisOS

# Initialize the system
genesis = GenesisOS()

# Start the system
await genesis.start()

# Get system status
status = genesis.get_status()
print(status)
```

2. Run the example:
```bash
python run_genesis.py
```

## Project Structure

```
genius_ai/
├── core/
│   ├── genesis.py          # Main Genesis OS implementation
│   ├── prism_x.py          # PRISM-X architecture
│   ├── evo_forge.py        # Evolutionary learning system
│   ├── planning.py         # Planning system
│   ├── imitation.py        # Imitation learning
│   ├── decision.py         # Decision engine
│   ├── memory.py           # Memory system
│   ├── vision.py           # Vision perception
│   ├── control.py          # Hardware control
│   └── brain.py            # Core brain functionality
├── tests/
│   └── test_genesis.py     # Test suite
├── config/
│   └── config.yaml         # Configuration file
├── requirements.txt        # Project dependencies
└── README.md              # Documentation
```

## Usage

1. Initialize the Genesis OS:
```python
from genius_ai.core.genesis import GenesisOS

# Create Genesis OS instance
genesis = GenesisOS()

# Start the system
genesis.start()
```

2. Monitor system status:
```python
# Get current status
status = genesis.get_status()
print(status)
```

3. Save and load system state:
```python
# Save current state
genesis.save_state("checkpoint.pth")

# Load saved state
genesis.load_state("checkpoint.pth")
```

## Testing

Run the test suite:
```bash
pytest tests/
```

For coverage report:
```bash
pytest --cov=genius_ai tests/
```

## Configuration

The system can be configured through `config/config.yaml`. Key configuration options include:

- Vision settings
- Memory parameters
- Decision engine configuration
- Resource management thresholds
- Learning parameters

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PRISM-X architecture for the base system design
- EVO-Forge for evolutionary learning capabilities
- Various open-source AI libraries and frameworks 