from typing import List, Tuple, NamedTuple


class Tile(NamedTuple):
    """Represents a tile: (m_start, k_start, n_start, m_size, k_size, n_size)"""
    m_start: int
    k_start: int
    n_start: int
    m_size: int
    k_size: int
    n_size: int
    
    def __repr__(self):
        return f"Tile(M[{self.m_start}:{self.m_start + self.m_size}], " \
               f"K[{self.k_start}:{self.k_start + self.k_size}], " \
               f"N[{self.n_start}:{self.n_start + self.n_size}])"


def tile_matmul(m: int, k: int, n: int, 
                tile_m: int = 256, 
                tile_k: int = 256, 
                tile_n: int = 256) -> List[Tile]:
    """
    Split matrix multiply C[M×N] = A[M×K] @ B[K×N] into tiles.
    
    Args:
        m, k, n: Matrix dimensions
        tile_m, tile_k, tile_n: Maximum tile sizes (default 256 for systolic array)
    
    Returns:
        List of tiles to process
    """
    tiles = []
    
    # Tile the M dimension
    for m_start in range(0, m, tile_m):
        m_size = min(tile_m, m - m_start)
        
        # Tile the N dimension
        for n_start in range(0, n, tile_n):
            n_size = min(tile_n, n - n_start)
            
            # Tile the K dimension (reduction axis)
            for k_start in range(0, k, tile_k):
                k_size = min(tile_k, k - k_start)
                
                tiles.append(Tile(
                    m_start=m_start,
                    k_start=k_start,
                    n_start=n_start,
                    m_size=m_size,
                    k_size=k_size,
                    n_size=n_size
                ))
    
    return tiles


def estimate_tile_bytes(tile: Tile) -> int:
    """
    Estimate memory footprint of a tile in bytes.
    Assumes 4 bytes per element (FP32).
    """
    a_bytes = tile.m_size * tile.k_size * 4
    b_bytes = tile.k_size * tile.n_size * 4
    c_bytes = tile.m_size * tile.n_size * 4
    return a_bytes + b_bytes + c_bytes
