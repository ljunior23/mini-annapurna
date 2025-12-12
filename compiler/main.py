import argparse
import pickle
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compiler.frontend import extract_from_shapes
from compiler.tiler import tile_matmul
from compiler.scheduler import schedule


def compile_matmul(m: int, k: int, n: int, output_file: str):
    """
    Compile matrix multiply into instruction sequence.
    
    Args:
        m, k, n: Matrix dimensions
        output_file: Path to save compiled program
    """
    print(f"Compiling {m}×{k} @ {k}×{n} matmul...")
    
    # Extract dimensions (in real case, would use FX graph)
    dims = extract_from_shapes(m, k, n)
    print(f"  Dimensions: M={dims[0]}, K={dims[1]}, N={dims[2]}")
    
    # Tile the computation
    tiles = tile_matmul(*dims)
    print(f"  Generated {len(tiles)} tiles")
    
    # Schedule and emit instructions
    instructions = schedule(tiles)
    print(f"  Emitted {len(instructions)} instructions")
    
    # Save to file
    with open(output_file, 'wb') as f:
        pickle.dump(instructions, f)
    
    print(f"  Saved to {output_file}")
    
    # Print first few instructions
    print("\nFirst 10 instructions:")
    for i, inst in enumerate(instructions[:10]):
        print(f"  {i:4d}: {inst}")
    
    return instructions


def main():
    parser = argparse.ArgumentParser(description='Mini-Annapurna Compiler')
    parser.add_argument('--m', type=int, required=True, help='M dimension')
    parser.add_argument('--k', type=int, required=True, help='K dimension')
    parser.add_argument('--n', type=int, required=True, help='N dimension')
    parser.add_argument('-o', '--output', type=str, default='prog.bin', 
                        help='Output file path')
    
    args = parser.parse_args()
    
    compile_matmul(args.m, args.k, args.n, args.output)


if __name__ == '__main__':
    main()
