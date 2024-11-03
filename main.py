import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utilities import ModelMemoryUtilities
from monitor import ModelMemoryMonitorGPU, ModelMemoryMonitorCPU  # Import both classes

def main(args):
    # Set device
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    # Automatically detect AMP support if using CUDA
    use_amp = device.type == "cuda" and torch.cuda.get_device_capability(0) >= (7, 0)

    # Initialize the appropriate monitor based on device
    if device.type == "cuda":
        monitor = ModelMemoryMonitorGPU(
            model_name=args.model_name,
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            torch_dtype=torch.float16 if use_amp else torch.float32,
            use_amp=use_amp,
            device=device.type
        )
    else:
        monitor = ModelMemoryMonitorCPU(
            model_name=args.model_name,
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            torch_dtype=torch.float32,
            device=device.type
        )

    # Prepare sample inputs or prompt based on the test
    sample_inputs = monitor.simulate_input_ids(args.max_seq_len)
    prompt = args.prompt or "Once upon a time"  # Default prompt for iterative inference if none provided

    # Run the specified test
    if args.test == "forward":
        if device.type == "cuda":
            prev_mem, peak_mem, cur_mem = monitor.test_cuda_forward_memory(sample_inputs, memory_unit=args.memory_unit)
        else:
            prev_mem, peak_mem, cur_mem = monitor.test_cpu_forward_memory(sample_inputs, memory_unit=args.memory_unit)
        print(f"Forward Memory Test - Prev: {prev_mem} {args.memory_unit.upper()}, Peak: {peak_mem} {args.memory_unit.upper()}, Current: {cur_mem} {args.memory_unit.upper()}")

    elif args.test == "iterative_inference":
        if device.type == "cuda":
            prev_mem, peak_mem_list, cur_mem_list = monitor.test_cuda_iterative_inference_memory(prompt, max_iters=args.max_iters, memory_unit=args.memory_unit)
        else:
            prev_mem, peak_mem_list, cur_mem_list = monitor.test_cpu_iterative_inference_memory(prompt, max_iters=args.max_iters, memory_unit=args.memory_unit)
        print(f"Iterative Inference Test - Initial: {prev_mem} {args.memory_unit.upper()}")
        print(f"Peak Memory per Iteration: {peak_mem_list}")
        print(f"Current Memory per Iteration: {cur_mem_list}")

    elif args.test == "training":
        if device.type == "cuda":
            memory_consumption = monitor.test_cuda_training_memory(max_iters=args.max_iters, memory_unit=args.memory_unit)
        else:
            memory_consumption = monitor.test_cpu_training_memory(max_iters=args.max_iters, memory_unit=args.memory_unit)
        print(f"Training Memory Consumption: {memory_consumption}")

    else:
        print("Invalid test selected. Choose from 'forward', 'iterative_inference', or 'training'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run memory monitoring tests for model inference and training.")
    
    # Model and device settings
    parser.add_argument("--model_name", type=str, required=True, help="Name or path of the model to load.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for the model input.")
    parser.add_argument("--max_seq_len", type=int, default=4096, help="Maximum sequence length for model input.")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu", help="Device to run the test on.")

    # Test selection and settings
    parser.add_argument("--test", type=str, choices=["forward", "iterative_inference", "training"], required=True, help="Type of memory test to run.")
    parser.add_argument("--memory_unit", type=str, choices=["byte", "mb", "gb"], default="mb", help="Unit for displaying memory usage.")
    parser.add_argument("--max_iters", type=int, default=100, help="Maximum iterations for iterative inference or training test.")
    parser.add_argument("--prompt", type=str, help="Prompt text for the iterative inference test.")

    args = parser.parse_args()
    main(args)
