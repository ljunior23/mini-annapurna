from typing import List, Dict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isa import Instruction, OpCode
from .memory import MemoryHierarchy
from .systolic import SystolicArray


class SimulatorCore:
    """
    Core simulator with scoreboard for hazard tracking.
    """
    
    def __init__(self):
        self.memory = MemoryHierarchy()
        self.systolic = SystolicArray()
        
        # Scoreboard: tracks pending operations on each scratchpad address
        self.scoreboard: Dict[int, int] = {}
        
        # Global cycle counter
        self.cycle = 0
        
        # Performance counters
        self.total_instructions = 0
        self.stall_cycles = 0
        self.bytes_transferred = 0
        
        # Trace data for visualization
        self.trace = []
    
    def tick(self, cycles: int = 1):
        """Advance simulation time"""
        self.cycle += cycles
        self.memory.tick(cycles)
        
        # Decrement all scoreboard entries
        for addr in list(self.scoreboard.keys()):
            self.scoreboard[addr] = max(0, self.scoreboard[addr] - cycles)
            if self.scoreboard[addr] == 0:
                del self.scoreboard[addr]
    
    def check_hazard(self, inst: Instruction) -> int:
        """
        Check for RAW hazards.
        
        Returns:
            Number of cycles to stall
        """
        stall = 0
        
        if inst.opcode == OpCode.LD:
            # Check if dst is being written
            stall = max(stall, self.scoreboard.get(inst.dst, 0))
        
        elif inst.opcode == OpCode.ST:
            # Check if src is being written
            stall = max(stall, self.scoreboard.get(inst.src0, 0))
        
        elif inst.opcode == OpCode.MAC:
            # Check if any operand is being written
            stall = max(stall, self.scoreboard.get(inst.src0, 0))
            stall = max(stall, self.scoreboard.get(inst.src1, 0))
            stall = max(stall, self.scoreboard.get(inst.dst, 0))
        
        elif inst.opcode == OpCode.SYNC:
            # Wait for all pending operations
            stall = max(self.scoreboard.values()) if self.scoreboard else 0
        
        return stall
    
    def execute(self, inst: Instruction):
        """Execute a single instruction"""
        self.total_instructions += 1
        
        if inst.opcode == OpCode.LD:
            # Load from DRAM to scratchpad
            # Assume 256×256 tile = 256KB (256*256*4 bytes)
            tile_bytes = 256 * 256 * 4
            cycles = self.memory.load(inst.src0, inst.dst, tile_bytes)
            self.scoreboard[inst.dst] = cycles
            self.bytes_transferred += tile_bytes
            self.tick(1)  # Issue latency
            
        elif inst.opcode == OpCode.ST:
            # Store from scratchpad to DRAM
            tile_bytes = 256 * 256 * 4
            cycles = self.memory.store(inst.src0, inst.dst, tile_bytes)
            self.bytes_transferred += tile_bytes
            self.tick(1)
            
        elif inst.opcode == OpCode.MAC:
            # Multiply-accumulate on systolic array
            # Assume 256×256×256 tile
            cycles, macs = self.systolic.compute_mac(256, 256, 256)
            self.scoreboard[inst.dst] = cycles
            self.tick(cycles)
            
        elif inst.opcode == OpCode.SYNC:
            # Already handled in check_hazard
            self.tick(1)
        
        # Record trace
        self.trace.append({
            'cycle': self.cycle,
            'opcode': inst.opcode.name,
            'dst': inst.dst,
            'src0': inst.src0,
            'src1': inst.src1
        })
    
    def run(self, program: List[Instruction]):
        """
        Run the entire program.
        """
        print(f"Starting simulation with {len(program)} instructions...")
        
        for i, inst in enumerate(program):
            if i % 100 == 0 and i > 0:
                print(f"  Progress: {i}/{len(program)} instructions")
            
            # Check for hazards and stall if needed
            stall = self.check_hazard(inst)
            if stall > 0:
                self.stall_cycles += stall
                self.tick(stall)
            
            # Execute instruction
            self.execute(inst)
        
        # Wait for all pending operations to complete
        if self.scoreboard:
            final_stall = max(self.scoreboard.values())
            self.tick(final_stall)
        
        print(f"Simulation complete!")
        self.print_stats()
    
    def print_stats(self):
        """Print simulation statistics"""
        print(f"\n{'='*60}")
        print(f"SIMULATION STATISTICS")
        print(f"{'='*60}")
        print(f"Total cycles:         {self.cycle:,}")
        print(f"Total instructions:   {self.total_instructions:,}")
        print(f"Stall cycles:         {self.stall_cycles:,}")
        print(f"Bytes transferred:    {self.bytes_transferred:,} ({self.bytes_transferred/1e6:.1f} MB)")
        
        # Systolic array stats
        sys_stats = self.systolic.get_stats()
        print(f"\nMAC utilization:      {sys_stats['utilization_pct']:.1f} %")
        print(f"Total MACs:           {sys_stats['total_macs']:,}")
        
        # Memory bandwidth efficiency
        if self.cycle > 0:
            bw_used = (self.bytes_transferred / self.cycle)  # bytes/cycle
            bw_max = self.memory.bandwidth_bytes_per_cycle
            bw_eff = (bw_used / bw_max) * 100.0
            print(f"Memory BW eff.:       {bw_eff:.1f} %")
            
            # FLOP/s (2 FLOPs per MAC)
            flops = sys_stats['total_macs'] * 2
            flops_per_sec = (flops / self.cycle) * 1e9  # Assume 1 GHz
            print(f"FLOP/s (simulated):   {flops_per_sec/1e9:.1f} GFLOP/s")
        
        print(f"{'='*60}\n")
    
    def save_trace(self, filename: str):
        """Save trace to CSV file"""
        import csv
        
        with open(filename, 'w', newline='') as f:
            if not self.trace:
                return
            
            writer = csv.DictWriter(f, fieldnames=self.trace[0].keys())
            writer.writeheader()
            writer.writerows(self.trace)
        
        print(f"Trace saved to {filename}")
