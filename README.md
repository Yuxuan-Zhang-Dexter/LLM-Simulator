
# LLM Memory Simulator

This repository provides tools for monitoring memory usage of large language models (LLMs) during inference and training on both CPU and GPU. The simulator is designed to help users track memory consumption and visualize memory trends across different stages of model usage.

## Setup Instructions

### 1. Set Up Virtual Environment

Create and activate a virtual environment named `llm-simulator-env`:

```bash
python -m venv llm-simulator-env
source llm-simulator-env/bin/activate  # On Windows, use `llm-simulator-env\Scripts\activate`
```

### 2. Install Dependencies

Install the required dependencies using `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3. Usage

The main script `main.py` supports memory monitoring for forward passes, training, and iterative inference on both CPU and GPU. Run the following commands to execute different tests.

#### Forward Memory Test (GPU)

```bash
python main.py --model_name "TinyLlama/TinyLlama-1.1B-step-50K-105b" --test forward --device cuda --max_seq_len 2048 --memory_unit gb
```

#### Training Memory Test (GPU)

```bash
python main.py --model_name "TinyLlama/TinyLlama-1.1B-step-50K-105b" --test training --device cuda --max_seq_len 2048 --max_iters 10 --memory_unit gb
```

#### Iterative Inference Memory Test (GPU)

```bash
python main.py --model_name "TinyLlama/TinyLlama-1.1B-step-50K-105b" --test iterative_inference --device cuda --max_seq_len 2048 --max_iters 50 --prompt "Once upon a time" --memory_unit gb
```

Replace `"cuda"` with `"cpu"` if running on a CPU, and adjust `--max_iters` or `--prompt` as needed.

### Project Structure

- `main.py`: Main script for running memory monitoring tests.
- `monitor.py`: Contains the `ModelMemoryMonitorCPU` and `ModelMemoryMonitorGPU` classes.
- `utilities.py`: Contains utility functions, including memory conversion and plotting.

### Requirements

Ensure you have `Python 3.7+` installed. All required Python packages are listed in `requirements.txt`.

### Deactivating the Environment

To exit the virtual environment, simply run:

```bash
deactivate
```

## License

This project is licensed under the MIT License.
