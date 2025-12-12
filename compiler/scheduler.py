from typing import List, Dict, Set
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isa import Instruction, OpCode
from compiler.tiler import Tile


class TileDAGNode:
    """Represents a tile computation in the dependency DAG"""
    def __init__(self, tile: Tile, tile_id: int):
        self.tile = tile
        self.tile_id = tile_id
        self.dependencies: Set[int] = set()
        self.a_addr = None  # Scratchpad address for A tile
        self.b_addr = None  # Scratchpad address for B tile
        self.c_addr = None  # Scratchpad address for C tile
    
    def add_dependency(self, dep_tile_id: int):
        """Add a dependency on another tile"""
        self.dependencies.add(dep_tile_id)


def build_dag(tiles: List[Tile]) -> List[TileDAGNode]:
    """
    Build dependency DAG for tiles.
    Tiles that write to the same C location depend on previous K-tiles.
    """
    nodes = []
    c_location_map: Dict[tuple, List[int]] = {}  # (m_start, n_start) -> [tile_ids]
    
    for i, tile in enumerate(tiles):
        node = TileDAGNode(tile, i)
        c_loc = (tile.m_start, tile.n_start)
        
        # Add dependency on previous k-tiles for same C location
        if c_loc in c_location_map:
            for prev_tile_id in c_location_map[c_loc]:
                prev_tile = tiles[prev_tile_id]
                if prev_tile.k_start < tile.k_start:
                    node.add_dependency(prev_tile_id)
        
        # Update c_location_map
        if c_loc not in c_location_map:
            c_location_map[c_loc] = []
        c_location_map[c_loc].append(i)
        
        nodes.append(node)
    
    return nodes


def allocate_scratchpad(nodes: List[TileDAGNode]):
    """
    Simple scratchpad allocator.
    Uses addresses 0-255 for A tiles, 256-511 for B tiles, 512-767 for C tiles.
    """
    for i, node in enumerate(nodes):
        # Rotate through scratchpad regions to avoid conflicts
        node.a_addr = (i * 3) % 256
        node.b_addr = 256 + (i * 3 + 1) % 256
        node.c_addr = 512 + (i * 3 + 2) % 256


def topological_sort(nodes: List[TileDAGNode]) -> List[TileDAGNode]:
    """
    Topological sort of DAG nodes.
    Simple implementation using DFS.
    """
    visited = set()
    sorted_nodes = []
    
    def visit(node_id: int):
        if node_id in visited:
            return
        visited.add(node_id)
        
        for dep_id in nodes[node_id].dependencies:
            visit(dep_id)
        
        sorted_nodes.append(nodes[node_id])
    
    for i in range(len(nodes)):
        visit(i)
    
    return sorted_nodes


def emit_instructions(nodes: List[TileDAGNode]) -> List[Instruction]:
    """
    Emit instructions for each tile in topological order.
    For each tile: LD A, LD B, SYNC, MAC, ST C
    """
    instructions = []
    
    for node in nodes:
        # Load A tile from DRAM
        instructions.append(Instruction(
            opcode=OpCode.LD,
            dst=node.a_addr,
            src0=node.tile_id * 3,  # DRAM address (simplified)
            src1=0
        ))
        
        # Load B tile from DRAM
        instructions.append(Instruction(
            opcode=OpCode.LD,
            dst=node.b_addr,
            src0=node.tile_id * 3 + 1,  # DRAM address (simplified)
            src1=0
        ))
        
        # Synchronization barrier
        instructions.append(Instruction(
            opcode=OpCode.SYNC,
            dst=0,
            src0=0,
            src1=0
        ))
        
        # MAC operation
        instructions.append(Instruction(
            opcode=OpCode.MAC,
            dst=node.c_addr,
            src0=node.a_addr,
            src1=node.b_addr
        ))
        
        # Store C tile to DRAM
        instructions.append(Instruction(
            opcode=OpCode.ST,
            dst=node.tile_id * 3 + 2,  # DRAM address (simplified)
            src0=node.c_addr,
            src1=0
        ))
    
    return instructions


def schedule(tiles: List[Tile]) -> List[Instruction]:
    """
    Main scheduling function: builds DAG and emits instructions.
    """
    nodes = build_dag(tiles)
    allocate_scratchpad(nodes)
    sorted_nodes = topological_sort(nodes)
    instructions = emit_instructions(sorted_nodes)
    return instructions
