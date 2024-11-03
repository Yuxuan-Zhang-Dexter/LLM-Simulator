from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
from typing import Literal, Tuple, Dict, Any
import psutil
from utilities import ModelMemoryUtilities
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp  = torch.cuda.is_available() and torch.cuda.get_device_capability(0) >= (7, 0)

class ModelMemoryMonitorGPU:
    def __init__(self, model_name, batch_size=1, max_seq_len=4096, torch_dtype=torch.float16, use_amp=False, device="cuda"):
        """
        Initialize ModelMemoryMonitor for tracking memory usage in inference and training.

        Args:
            model_name (str): Model name or path.
            batch_size (int): Number of samples in a batch.
            max_seq_len (int): Maximum sequence length.
            torch_dtype (torch.dtype): Data type (e.g., torch.float16).
            use_amp (bool): If True, enables mixed precision inference.
            device (str): Device for inference ('cuda' or 'cpu').
        """
        self.model_name = model_name
        self.torch_dtype = torch_dtype
        self.device = device
        self.use_amp = use_amp
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=self.torch_dtype).to(device)

        # Set pad_token to eos_token if no padding token is available
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def simulate_input_ids(self, sequence_length: int, only_padding=False):
        """
        Generate dummy input IDs for a given sequence length.

        Args:
            sequence_length (int): Target sequence length.
            only_padding (bool): If True, generate only padding tokens.

        Returns:
            dict: Input IDs and attention masks.
        """
        dummy_text = "" if only_padding else " ".join(["token"] * int(sequence_length * 1.5))
        inputs = self.tokenizer(dummy_text, return_tensors="pt", padding=True, truncation=True, max_length=self.max_seq_len)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        
        actual_length = inputs["input_ids"].shape[1]
        if actual_length != sequence_length:
            print(f"Warning: Expected sequence length ({sequence_length}) does not match actual input length ({actual_length}).")

        attention_mask_sum = inputs["attention_mask"].sum().item()
        if attention_mask_sum != sequence_length:
            print(f"Warning: Attention mask sum ({attention_mask_sum}) does not match expected sequence length ({sequence_length}).")

        return inputs

    def test_cuda_forward_memory(self, sample_inputs, memory_unit: Literal['byte', 'mb', 'gb'] = 'byte') -> Tuple[float, float, float]:
        """
        Measure memory usage during a forward pass.

        Args:
            sample_inputs (dict): Model input data.
            memory_unit (str): Unit for memory display.

        Returns:
            tuple: Previous, peak, and current memory in specified unit.
        """
        self.model.to("cpu")  # Move model to CPU to measure existing memory reliably
        exist_memory = torch.cuda.memory_allocated(self.device)
        self.model.to(self.device)  # Move model back to GPU
        self.model.eval()

        prev_memory = torch.cuda.memory_allocated(self.device)
        torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            output = self.model(**sample_inputs)
            output_sum = ModelMemoryUtilities.get_logits(output).sum()
        
        peak_memory = torch.cuda.max_memory_allocated(self.device) - exist_memory
        cur_memory = torch.cuda.memory_allocated(self.device) - exist_memory

        prev_memory = ModelMemoryUtilities.convert_memory(prev_memory - exist_memory, memory_unit)
        peak_memory = ModelMemoryUtilities.convert_memory(peak_memory, memory_unit)
        cur_memory = ModelMemoryUtilities.convert_memory(cur_memory, memory_unit)

        print(f"Previous Memory: {prev_memory:.2f} {memory_unit.upper()}; Peak Memory: {peak_memory:.2f} {memory_unit.upper()}; Current Memory: {cur_memory:.2f} {memory_unit.upper()}")

        return prev_memory, peak_memory, cur_memory
    
    def test_cuda_iterative_inference_memory(self, prompt: str, max_iters: int = 100, memory_unit: Literal['byte', 'mb', 'gb'] = 'byte') -> Tuple[float, list, list]:
        """
        Estimate memory usage during iterative token-by-token generation.

        Args:
            prompt (str): Initial prompt for generation.
            max_iters (int): Maximum number of iterations (tokens) to generate.
            memory_unit (Literal['byte', 'mb', 'gb']): Unit for memory measurement ('byte', 'mb', 'gb').

        Returns:
            Tuple[float, list, list]: Initial memory, list of peak memory per iteration, list of current memory per iteration.
        """
        # Move model to CPU to measure existing memory reliably
        self.model.to("cpu")
        exist_memory = torch.cuda.memory_allocated(self.device)
        self.model.to(self.device)
        self.model.eval()

        # Initial memory state before starting the generation loop
        prev_memory = torch.cuda.memory_allocated(self.device)
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=self.max_seq_len)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Lists to track memory usage across iterations
        peak_memory_lst, cur_memory_lst = [], []

        for i in range(max_iters):
            torch.cuda.reset_peak_memory_stats()

            with torch.no_grad():
                if self.use_amp:
                    with torch.amp.autocast(device_type=str(self.device)):
                        outputs = self.model.generate(
                            input_ids,
                            attention_mask=attention_mask,
                            max_length=input_ids.shape[1] + 1,
                            eos_token_id=self.tokenizer.eos_token_id,
                            pad_token_id=self.tokenizer.eos_token_id,
                            do_sample=False
                        )
                else:
                    outputs = self.model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_length=input_ids.shape[1] + 1,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.eos_token_id,
                        do_sample=False
                    )

                next_token_id = outputs[:, -1:]
                input_ids = torch.cat([input_ids, next_token_id], dim=1)

                # Update attention mask to include the new token
                new_attention_mask = torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=self.device)
                attention_mask = torch.cat([attention_mask, new_attention_mask], dim=1)

                # Stop if EOS token is generated
                if next_token_id.item() == self.tokenizer.eos_token_id:
                    print(f"EOS token generated at iteration {i+1}")
                    break

            # Record peak and current memory usage
            peak_memory_lst.append(torch.cuda.max_memory_allocated(self.device))
            cur_memory_lst.append(torch.cuda.memory_allocated(self.device))

        # Decode generated text
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        print(f'Generated Text: {generated_text}')

        # Adjust for pre-existing memory
        prev_memory -= exist_memory
        peak_memory_lst = [peak_memory - exist_memory for peak_memory in peak_memory_lst]
        cur_memory_lst = [cur_memory - exist_memory for cur_memory in cur_memory_lst]

        # Convert memory values to the specified unit
        prev_memory = ModelMemoryUtilities.convert_memory(prev_memory, memory_unit)
        peak_memory_lst = [ModelMemoryUtilities.convert_memory(mem, memory_unit) for mem in peak_memory_lst]
        cur_memory_lst = [ModelMemoryUtilities.convert_memory(mem, memory_unit) for mem in cur_memory_lst]

        return prev_memory, peak_memory_lst, cur_memory_lst

    def test_cuda_training_memory(self, optimizer_type=torch.optim.Adam, max_iters = 1, memory_unit: Literal['byte', 'mb', 'gb'] = 'byte' ):
        device = self.device
        self.model.cpu()
        torch.cuda.reset_peak_memory_stats()
        exist_memory = torch.cuda.memory_allocated(device)
        self.model.to(device)
        self.model.train()
        pre_memory = torch.cuda.memory_allocated(device)

        sample_inputs = self.simulate_input_ids(self.max_seq_len)
        optimizer = optimizer_type(self.model.parameters(), lr=.001)

        optimizer.zero_grad()

        peak_memory_dict = {
            'forward': [],
            'backward': [],
            'optimizer': []
            }

        memory_dict =  {
            'model_loading': None, 
            'forward_pass': [], 
            'backward_pass': [], 
            'optimize_model': []
            }

        for i in range(max_iters):
            if use_amp:
                with torch.amp.autocast(device_type=str(self.device)):
                    output = self.model(**sample_inputs)
                    output_logits = ModelMemoryUtilities.get_logits(output)
                    output_logits_sum = output_logits.sum()
                    forward_memory = torch.cuda.memory_allocated(device)
                    forward_peak_memory = torch.cuda.max_memory_allocated(device)
                    torch.cuda.reset_peak_memory_stats()

                output_logits_sum.backward()
                backward_memory = torch.cuda.memory_allocated(device)
                backward_peak_memory = torch.cuda.max_memory_allocated(device)
                torch.cuda.reset_peak_memory_stats()

                optimizer.step()
                optimizer_memory = torch.cuda.memory_allocated(device)
                optimizer_peak_memory = torch.cuda.max_memory_allocated(device)

            else:
                output = self.model(**sample_inputs).sum()
                output_logits = ModelMemoryUtilities.get_logits(output)
                output_logits_sum = output_logits.sum()
                forward_memory = torch.cuda.memory_allocated(self.device)
                forward_peak_memory = torch.cuda.max_memory_allocated(self.device)
                torch.cuda.reset_peak_memory_stats()

                output_logits_sum.backward()
                backward_memory = torch.cuda.memory_allocated(self.device)
                backward_peak_memory = torch.cuda.max_memory_allocated(self.device)
                torch.cuda.reset_peak_memory_stats()

                optimizer.step()
                optimizer_memory = torch.cuda.memory_allocated(self.device)
                optimizer_peak_memory = torch.cuda.max_memory_allocated(self.device)

            pre_memory -= exist_memory
            forward_memory -= exist_memory
            forward_peak_memory -= exist_memory
            backward_memory -= exist_memory
            backward_peak_memory -= exist_memory
            optimizer_memory -= exist_memory
            optimizer_peak_memory -= exist_memory

            peak_memory_dict['forward'].append(forward_peak_memory)
            peak_memory_dict['backward'].append(backward_peak_memory)
            peak_memory_dict['optimizer'].append(optimizer_peak_memory)

            memory_dict['forward_pass'].append(forward_memory)
            memory_dict['backward_pass'].append(backward_memory)
            memory_dict['optimize_model'].append(optimizer_memory)

        memory_dict['model_loading'] = pre_memory
        filter_peak_memory_dict = {key: max(value) for key, value in peak_memory_dict.items()}
        max_peak_stage, max_peak_memory = max(filter_peak_memory_dict.items(), key=lambda x: x[1])
        memory_dict[f'max_peak_memory({max_peak_stage})'] = max_peak_memory
        
        print(f"The training max peak memory: {ModelMemoryUtilities.convert_memory(max_peak_memory, memory_unit)} ({max_peak_stage} stage)")
        converted_memory_dict = {
            key: [ModelMemoryUtilities.convert_memory(element, memory_unit) for element in value] if isinstance(value, list) else ModelMemoryUtilities.convert_memory(value, memory_unit)
            for key, value in memory_dict.items()
        }
        print(f"the training memory consumption: {converted_memory_dict}")

        return  converted_memory_dict
    



class ModelMemoryMonitorCPU:
    def __init__(self, model_name, batch_size=1, max_seq_len=4096, torch_dtype=torch.float32, device="cpu"):
        """
        Initialize ModelMemoryMonitorCPU for tracking CPU memory usage in inference and training.

        Args:
            model_name (str): Model name or path.
            batch_size (int): Number of samples in a batch.
            max_seq_len (int): Maximum sequence length.
            torch_dtype (torch.dtype): Data type (e.g., torch.float32).
            device (str): Device for inference ('cpu').
        """
        self.model_name = model_name
        self.torch_dtype = torch_dtype
        self.device = device
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=self.torch_dtype).to(device)

        # Set pad_token to eos_token if no padding token is available
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def simulate_input_ids(self, sequence_length: int, only_padding=False):
        """
        Generate dummy input IDs for a given sequence length.

        Args:
            sequence_length (int): Target sequence length.
            only_padding (bool): If True, generate only padding tokens.

        Returns:
            dict: Input IDs and attention masks.
        """
        dummy_text = "" if only_padding else " ".join(["token"] * int(sequence_length * 1.5))
        inputs = self.tokenizer(dummy_text, return_tensors="pt", padding=True, truncation=True, max_length=self.max_seq_len)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        
        actual_length = inputs["input_ids"].shape[1]
        if actual_length != sequence_length:
            print(f"Warning: Expected sequence length ({sequence_length}) does not match actual input length ({actual_length}).")

        attention_mask_sum = inputs["attention_mask"].sum().item()
        if attention_mask_sum != sequence_length:
            print(f"Warning: Attention mask sum ({attention_mask_sum}) does not match expected sequence length ({sequence_length}).")

        return inputs


    def test_cpu_forward_memory(self, sample_inputs, memory_unit: Literal['byte', 'mb', 'gb'] = 'byte') -> Tuple[float, float, float]:
        """
        Measure memory usage during a forward pass on CPU.

        Args:
            sample_inputs (dict): Model input data.
            memory_unit (str): Unit for memory display.

        Returns:
            tuple: Previous, peak, and current memory in specified unit.
        """
        # Initial memory usage
        self.model.cpu()
        prev_memory = ModelMemoryUtilities.measure_cpu_memory()

        # Perform forward pass and measure memory usage
        with torch.no_grad():
            output = self.model(**sample_inputs)
            output_sum = ModelMemoryUtilities.get_logits(output).sum()

        cur_memory = ModelMemoryUtilities.measure_cpu_memory()
        peak_memory = max(prev_memory, cur_memory)  # Peak CPU memory usage (approximate)

        # Convert memory values to the specified unit
        prev_memory = ModelMemoryUtilities.convert_memory(prev_memory, memory_unit)
        peak_memory = ModelMemoryUtilities.convert_memory(peak_memory, memory_unit)
        cur_memory = ModelMemoryUtilities.convert_memory(cur_memory, memory_unit)

        print(f"Previous Memory: {prev_memory:.2f} {memory_unit.upper()}; Peak Memory: {peak_memory:.2f} {memory_unit.upper()}; Current Memory: {cur_memory:.2f} {memory_unit.upper()}")
        return prev_memory, peak_memory, cur_memory

    def test_cpu_iterative_inference_memory(self, prompt: str, max_iters: int = 100, memory_unit: Literal['byte', 'mb', 'gb'] = 'byte') -> Tuple[float, list[float], list[float]]:
        """
        Estimate memory usage during iterative token-by-token generation on CPU.

        Args:
            prompt (str): Initial prompt for generation.
            max_iters (int): Maximum number of iterations (tokens) to generate.
            memory_unit (Literal['byte', 'mb', 'gb']): Unit for memory measurement ('byte', 'mb', 'gb').

        Returns:
            Tuple[float, list, list]: Initial memory, list of peak memory per iteration, list of current memory per iteration.
        """
        self.model.cpu()
        prev_memory = ModelMemoryUtilities.measure_cpu_memory()
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=self.max_seq_len)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Lists to track memory usage across iterations
        peak_memory_lst, cur_memory_lst = [], []

        for i in range(max_iters):
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=input_ids.shape[1] + 1,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=False
                )

                next_token_id = outputs[:, -1:]
                input_ids = torch.cat([input_ids, next_token_id], dim=1)

                # Update attention mask to include the new token
                new_attention_mask = torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=self.device)
                attention_mask = torch.cat([attention_mask, new_attention_mask], dim=1)

                # Stop if EOS token is generated
                if next_token_id.item() == self.tokenizer.eos_token_id:
                    print(f"EOS token generated at iteration {i+1}")
                    break

            # Record memory usage
            peak_memory = max(prev_memory, ModelMemoryUtilities.measure_cpu_memory())
            cur_memory = ModelMemoryUtilities.measure_cpu_memory()

            peak_memory_lst.append(ModelMemoryUtilities.convert_memory(peak_memory, memory_unit))
            cur_memory_lst.append(ModelMemoryUtilities.convert_memory(cur_memory, memory_unit))

        return prev_memory, peak_memory_lst, cur_memory_lst

    def test_cpu_training_memory(self, optimizer_type=torch.optim.Adam, max_iters=1, memory_unit: Literal['byte', 'mb', 'gb'] = 'byte') -> Dict[str, Any]:
        """
        Measure memory usage during training on CPU.

        Args:
            optimizer_type: Optimizer class (default: Adam).
            max_iters (int): Number of iterations.
            memory_unit (str): Unit for memory display.

        Returns:
            dict: Memory consumption statistics.
        """
        self.model.cpu()
        prev_memory = ModelMemoryUtilities.measure_cpu_memory()
        self.model.train()
        sample_inputs = self.simulate_input_ids(self.max_seq_len)
        optimizer = optimizer_type(self.model.parameters(), lr=0.001)

        peak_memory_dict = {
            'forward': [],
            'backward': [],
            'optimizer': []
        }
        memory_dict = {
            'model_loading': prev_memory,
            'forward_pass': [],
            'backward_pass': [],
            'optimize_model': []
        }

        for i in range(max_iters):
            optimizer.zero_grad()
            
            # Forward pass
            output = self.model(**sample_inputs)
            output_logits = ModelMemoryUtilities.get_logits(output)
            output_logits_sum = output_logits.sum()
            forward_memory = ModelMemoryUtilities.measure_cpu_memory()
            peak_memory_dict['forward'].append(forward_memory)
            memory_dict['forward_pass'].append(forward_memory)

            # Backward pass
            output_logits_sum.backward()
            backward_memory = ModelMemoryUtilities.measure_cpu_memory()
            peak_memory_dict['backward'].append(backward_memory)
            memory_dict['backward_pass'].append(backward_memory)

            # Optimizer step
            optimizer.step()
            optimizer_memory = ModelMemoryUtilities.measure_cpu_memory()
            peak_memory_dict['optimizer'].append(optimizer_memory)
            memory_dict['optimize_model'].append(optimizer_memory)

        # Calculate maximum peak memory across stages
        filter_peak_memory_dict = {key: max(value) for key, value in peak_memory_dict.items()}
        max_peak_stage, max_peak_memory = max(filter_peak_memory_dict.items(), key=lambda x: x[1])

        memory_dict[f'max_peak_memory({max_peak_stage})'] = max_peak_memory

        # Convert all memory values to the specified unit
        converted_memory_dict = {
            key: [ModelMemoryUtilities.convert_memory(value, memory_unit) for value in values] if isinstance(values, list) else ModelMemoryUtilities.convert_memory(values, memory_unit)
            for key, values in memory_dict.items()
        }
        print(f"the training memory consumption: {converted_memory_dict}")
        return converted_memory_dict