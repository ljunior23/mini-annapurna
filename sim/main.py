"""
Simulator main entry point
"""
import argparse
import pickle
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.core import SimulatorCore


def simulate(program_file: str, trace_file: str = None):
    """
    Simulate a compiled program.
    
    Args:
        program_file: Path to compiled program
        trace_file: Optional path to save execution trace
    """
    # Load program
    print(f"Loading program from {program_file}...")
    with open(program_file, 'rb') as f:
        program = pickle.load(f)
    
    print(f"Loaded {len(program)} instructions")
    
    # Create simulator
    sim = SimulatorCore()
    
    # Run simulation
    sim.run(program)
    
    # Save trace if requested
    if trace_file:
        sim.save_trace(trace_file)
    
    return sim


def main():
    parser = argparse.ArgumentParser(description='Mini-Annapurna Simulator')
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Input program file')
    parser.add_argument('-t', '--trace', type=str, default=None,
                        help='Output trace file (CSV)')
    
    args = parser.parse_args()
    
    simulate(args.input, args.trace)


if __name__ == '__main__':
    main()
