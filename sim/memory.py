from typing import Dict


class MemoryHierarchy:
    """
    Models bandwidth-limited memory hierarchy.
    Uses token-bucket for bandwidth limiting.
    """
    
    def __init__(self, 
                 bandwidth_gbps: float = 100.0,
                 dram_latency_cycles: int = 100,
                 l1_latency_cycles: int = 1):
        """
        Args:
            bandwidth_gbps: Die-edge bandwidth in GB/s
            dram_latency_cycles: DRAM access latency
            l1_latency_cycles: L1/scratchpad access latency
        """
        self.bandwidth_gbps = bandwidth_gbps
        self.bandwidth_bytes_per_cycle = bandwidth_gbps * 1e9 / 1e9  # Assume 1 GHz
        self.dram_latency = dram_latency_cycles
        self.l1_latency = l1_latency_cycles
        
        # Token bucket for bandwidth limiting
        self.tokens = 0.0
        self.max_tokens = bandwidth_gbps * 10  # Burst capacity
        
        # Memory storage (simplified)
        self.dram: Dict[int, float] = {}
        self.scratchpad: Dict[int, float] = {}
    
    def tick(self, cycles: int = 1):
        """
        Advance time and refill bandwidth tokens.
        """
        self.tokens = min(self.max_tokens, 
                         self.tokens + self.bandwidth_bytes_per_cycle * cycles)
    
    def load(self, dram_addr: int, scratch_addr: int, num_bytes: int) -> int:
        """
        Load data from DRAM to scratchpad.
        
        Returns:
            Number of cycles required
        """
        # Check bandwidth availability
        cycles_needed = max(1, int(num_bytes / self.bandwidth_bytes_per_cycle))
        
        if self.tokens >= num_bytes:
            self.tokens -= num_bytes
        else:
            # Need to wait for bandwidth
            wait_cycles = int((num_bytes - self.tokens) / self.bandwidth_bytes_per_cycle)
            self.tick(wait_cycles)
            cycles_needed += wait_cycles
            self.tokens -= num_bytes
        
        # Add DRAM latency
        total_cycles = cycles_needed + self.dram_latency
        
        # Perform the transfer (simplified - just mark as loaded)
        self.scratchpad[scratch_addr] = self.dram.get(dram_addr, 0.0)
        
        return total_cycles
    
    def store(self, scratch_addr: int, dram_addr: int, num_bytes: int) -> int:
        """
        Store data from scratchpad to DRAM.
        
        Returns:
            Number of cycles required
        """
        # Similar to load
        cycles_needed = max(1, int(num_bytes / self.bandwidth_bytes_per_cycle))
        
        if self.tokens >= num_bytes:
            self.tokens -= num_bytes
        else:
            wait_cycles = int((num_bytes - self.tokens) / self.bandwidth_bytes_per_cycle)
            self.tick(wait_cycles)
            cycles_needed += wait_cycles
            self.tokens -= num_bytes
        
        total_cycles = cycles_needed + self.dram_latency
        
        # Perform the transfer
        self.dram[dram_addr] = self.scratchpad.get(scratch_addr, 0.0)
        
        return total_cycles
    
    def get_stats(self) -> Dict[str, float]:
        """Return memory statistics"""
        return {
            'bandwidth_gbps': self.bandwidth_gbps,
            'current_tokens': self.tokens
        }
