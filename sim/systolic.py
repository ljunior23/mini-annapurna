"""
Systolic array: 256×256 grid with 2-cycle pipeline latency
"""
import numpy as np
from typing import Tuple


class SystolicArray:
    """
    256×256 systolic array for matrix multiplication.
    Models pipeline latency and utilization.
    """
    
    def __init__(self, rows: int = 256, cols: int = 256, pipeline_latency: int = 2):
        """
        Args:
            rows: Number of rows in systolic array
            cols: Number of columns in systolic array
            pipeline_latency: Cycles for data to flow through pipeline
        """
        self.rows = rows
        self.cols = cols
        self.pipeline_latency = pipeline_latency
        
        # Performance counters
        self.total_macs = 0
        self.total_mac_cycles = 0
        self.idle_cycles = 0
    
    def compute_mac(self, m: int, k: int, n: int) -> Tuple[int, int]:
        """
        Compute MAC operation for tile of size (m, k, n).
        
        Returns:
            (cycles, num_macs) tuple
        """
        # Number of MACs = m * k * n
        num_macs = m * k * n
        
        # Systolic array can process min(m, rows) × min(n, cols) in parallel
        parallel_m = min(m, self.rows)
        parallel_n = min(n, self.cols)
        
        # Number of steps needed for K dimension
        k_steps = k
        
        # Total cycles = pipeline_latency + k_steps (for data flow)
        # Need to process ceiling(m/rows) × ceiling(n/cols) tiles sequentially
        m_batches = (m + self.rows - 1) // self.rows
        n_batches = (n + self.cols - 1) // self.cols
        
        cycles_per_tile = self.pipeline_latency + k_steps
        total_cycles = cycles_per_tile * m_batches * n_batches
        
        # Update counters
        self.total_macs += num_macs
        self.total_mac_cycles += total_cycles
        
        return total_cycles, num_macs
    
    def get_utilization(self) -> float:
        """
        Calculate MAC utilization as percentage.
        Utilization = actual MACs / (array_size × cycles)
        """
        if self.total_mac_cycles == 0:
            return 0.0
        
        max_possible_macs = self.rows * self.cols * self.total_mac_cycles
        return (self.total_macs / max_possible_macs) * 100.0
    
    def get_stats(self) -> dict:
        """Return performance statistics"""
        return {
            'total_macs': self.total_macs,
            'total_mac_cycles': self.total_mac_cycles,
            'utilization_pct': self.get_utilization(),
            'array_size': f"{self.rows}×{self.cols}",
        }
    
    def reset_stats(self):
        """Reset performance counters"""
        self.total_macs = 0
        self.total_mac_cycles = 0
        self.idle_cycles = 0
