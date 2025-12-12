"""
Mini-Annapurna ISA Definition
32-bit instruction format: | 2b opcode | 10b dst | 10b src0 | 10b src1 |
"""
from enum import IntEnum
from typing import NamedTuple


class OpCode(IntEnum):
    """4-op ISA for systolic array"""
    LD = 0b00      # Load: DRAM → scratchpad
    ST = 0b01      # Store: scratchpad → DRAM
    MAC = 0b10     # Multiply-accumulate on systolic array
    SYNC = 0b11    # Synchronization barrier


class Instruction(NamedTuple):
    """32-bit instruction"""
    opcode: OpCode
    dst: int       # 10-bit destination address
    src0: int      # 10-bit source address 0
    src1: int      # 10-bit source address 1
    
    def encode(self) -> int:
        """Pack instruction into 32-bit integer"""
        assert 0 <= self.dst < 1024, "dst must be 10-bit"
        assert 0 <= self.src0 < 1024, "src0 must be 10-bit"
        assert 0 <= self.src1 < 1024, "src1 must be 10-bit"
        
        encoded = (int(self.opcode) << 30) | (self.dst << 20) | (self.src0 << 10) | self.src1
        return encoded
    
    @staticmethod
    def decode(encoded: int) -> 'Instruction':
        """Unpack 32-bit integer into instruction"""
        opcode = OpCode((encoded >> 30) & 0b11)
        dst = (encoded >> 20) & 0x3FF
        src0 = (encoded >> 10) & 0x3FF
        src1 = encoded & 0x3FF
        return Instruction(opcode, dst, src0, src1)
    
    def __str__(self) -> str:
        if self.opcode == OpCode.LD:
            return f"LD   [{self.dst:4d}] ← DRAM[{self.src0:4d}]"
        elif self.opcode == OpCode.ST:
            return f"ST   DRAM[{self.dst:4d}] ← [{self.src0:4d}]"
        elif self.opcode == OpCode.MAC:
            return f"MAC  C[{self.dst:4d}] += A[{self.src0:4d}] * B[{self.src1:4d}]"
        else:
            return f"SYNC"


# Memory layout constants
TILE_ALIGNMENT = 64  # bytes
SCRATCHPAD_SIZE = 1024  # entries (10-bit addressable)
