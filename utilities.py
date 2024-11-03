# Packages
import torch
from typing import Literal, Dict, Any
import matplotlib.pyplot as plt
import psutil


class ModelMemoryUtilities:
    @staticmethod
    def convert_memory(memory: int, memory_unit: Literal['byte', 'mb', 'gb'] = 'byte') -> float:
        """Convert memory to the specified unit."""
        if memory_unit == 'mb':
            return memory / 1048576  # Convert bytes to MB
        elif memory_unit == 'gb':
            return memory / 1073741824  # Convert bytes to GB
        return memory  # Default to bytes
    
    @staticmethod
    def measure_cpu_memory() -> int:
        """Measure CPU memory usage in bytes."""
        process = psutil.Process()
        return process.memory_info().rss

    @staticmethod
    def get_logits(model_output) -> torch.Tensor:
        """Extract logits from model output, supporting various output formats."""
        if hasattr(model_output, 'logits'):
            return model_output.logits
        elif isinstance(model_output, torch.Tensor):
            return model_output
        else:
            raise ValueError("Model output does not contain logits or is not a tensor.")
        
    @staticmethod
    def measure_cpu_memory():
        """Measure CPU memory usage in bytes."""
        return psutil.Process().memory_info().rss

    @staticmethod
    def draw_memory_lines(prev_memory: int, cur_memory_lst: list, peak_memory_lst: list = None, memory_unit: Literal['byte', 'mb', 'gb'] = 'byte', filename = None):
        """Plot memory usage over iterations."""
        conv_prev_memory = ModelMemoryUtilities.convert_memory(prev_memory, memory_unit)
        if peak_memory_lst is not None:
            conv_peak_memory = [ModelMemoryUtilities.convert_memory(peak, memory_unit) for peak in peak_memory_lst]
        conv_cur_memory = [ModelMemoryUtilities.convert_memory(cur, memory_unit) for cur in cur_memory_lst]

        plt.figure(figsize=(12, 6))
        iterations = range(len(cur_memory_lst))

        # Baseline memory line
        plt.axhline(y=conv_prev_memory, color='gray', linestyle='--', linewidth=1.5, label=f'Baseline ({conv_prev_memory:.2f} {memory_unit.upper()})')

        # Peak and current memory lines
        if peak_memory_lst is not None:
            plt.plot(iterations, conv_peak_memory, label='Peak Memory', color='#FF5733', linewidth=2.5, linestyle='-', marker='o', markersize=5, alpha=0.85)
        plt.plot(iterations, conv_cur_memory, label='Current Memory', color='#3498DB', linewidth=2.5, linestyle='-', marker='x', markersize=5, alpha=0.85)

        # Labels and title
        plt.xlabel("Iterations", fontsize=12)
        plt.ylabel(f"Memory Usage ({memory_unit.upper()})", fontsize=12)
        plt.title("Memory Usage per Iteration", fontsize=16, fontweight='bold')
        plt.legend()
        plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)

        plt.tight_layout()
        # Save the figure if filename is provided
        if filename:
            plt.savefig(filename)
        plt.show()
        
    @staticmethod
    def draw_memory_from_dict(memory_dict: Dict[str, Any], memory_unit: Literal['byte', 'mb', 'gb'] = 'gb', filename = None):
        """
        Draw a memory usage plot with fixed lines for 'model_loading' and 'max_peak_memory', and dynamic lines
        for each stage list found in the dictionary.
        
        :param memory_dict: Dictionary containing memory usage for different stages and fixed values.
        :param memory_unit: The unit for memory display ('byte', 'mb', 'gb').
        """
        
        plt.figure(figsize=(12, 6))

        # Extract and plot fixed lines for 'model_loading' and any 'max_peak_memory'
        fixed_lines = {key: ModelMemoryUtilities.convert_memory(value, memory_unit) for key, value in memory_dict.items()
                       if 'max_peak_memory' in key or key == 'model_loading'}
        for name, value in fixed_lines.items():
            plt.axhline(y=value, linestyle='--', linewidth=1.5, label=f"{name} ({value:.2f} {memory_unit.upper()})")

        # Plot dynamic lines for each memory stage list (e.g., forward_pass, backward_pass, optimize_model)
        dynamic_lines = {key: [ModelMemoryUtilities.convert_memory(val, memory_unit) for val in values] for key, values in memory_dict.items() if isinstance(values, list)}
        iterations = range(len(next(iter(dynamic_lines.values()))))  # Define iterations based on list length

        for stage_name, memory_values in dynamic_lines.items():
            plt.plot(iterations, memory_values, label=f'{stage_name.capitalize()} Memory', marker='o', markersize=5, alpha=0.85)

        # Labels and title
        plt.xlabel("Iterations", fontsize=12)
        plt.ylabel(f"Memory Usage ({memory_unit.upper()})", fontsize=12)
        plt.title("Memory Consumption per Iteration by Stage", fontsize=16, fontweight='bold')
        plt.legend()
        plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)
        
        plt.tight_layout()

        # Save the figure if filename is provided
        if filename:
            plt.savefig(filename)

        plt.show()
