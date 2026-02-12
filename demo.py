import os
import sys

def demo_512x512():
    """Demo 512×512 matmul"""
    print("="*70)
    print("MINI-ANNAPURNA DEMO: 512×512 Matrix Multiplication")
    print("="*70)
    print()
    
    # Step 1: Compile
    print("STEP 1: Compiling...")
    print("-" * 70)
    from compiler.main import compile_matmul
    compile_matmul(512, 512, 512, 'demo_prog.bin')
    print()
    
    # Step 2: Simulate
    print("STEP 2: Simulating...")
    print("-" * 70)
    from sim.main import simulate
    sim = simulate('demo_prog.bin', 'demo_trace.csv')
    print()
    
    # Step 3: Visualize
    print("STEP 3: Generating Roofline Plot...")
    print("-" * 70)
    from viz.roofline import plot_roofline
    plot_roofline('demo_trace.csv', 'demo_roofline.png')
    print()
    
    # Summary
    print("="*70)
    print("DEMO COMPLETE!")
    print("="*70)
    print(f"Generated files:")
    print(f"  - demo_prog.bin      (compiled program)")
    print(f"  - demo_trace.csv     (execution trace)")
    print(f"  - demo_roofline.png  (roofline plot)")
    print()
    print("View the roofline plot to see performance characteristics!")


def demo_comparison():
    """Compare different matrix sizes"""
    print("="*70)
    print("MINI-ANNAPURNA: Multi-Size Comparison")
    print("="*70)
    print()
    
    sizes = [
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024)
    ]
    
    from compiler.main import compile_matmul
    from sim.main import simulate
    
    results = []
    
    for m, k, n in sizes:
        print(f"\nTesting {m}×{k}×{n}...")
        print("-" * 70)
        
        prog_file = f'prog_{m}x{k}x{n}.bin'
        
        # Compile
        compile_matmul(m, k, n, prog_file)
        
        # Simulate
        sim = simulate(prog_file)
        
        results.append({
            'size': f'{m}×{k}×{n}',
            'cycles': sim.cycle,
            'utilization': sim.systolic.get_utilization(),
            'instructions': sim.total_instructions
        })
        
        # Cleanup
        os.unlink(prog_file)
    
    # Print comparison table
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    print(f"{'Size':<15} {'Cycles':<15} {'Utilization':<15} {'Instructions':<15}")
    print("-" * 70)
    for r in results:
        print(f"{r['size']:<15} {r['cycles']:<15,} {r['utilization']:<14.1f}% {r['instructions']:<15,}")


def show_isa_examples():
    """Show ISA instruction examples"""
    print("="*70)
    print("MINI-ANNAPURNA ISA EXAMPLES")
    print("="*70)
    print()
    
    from isa import Instruction, OpCode
    
    examples = [
        Instruction(OpCode.LD, 100, 500, 0),
        Instruction(OpCode.LD, 256, 501, 0),
        Instruction(OpCode.SYNC, 0, 0, 0),
        Instruction(OpCode.MAC, 512, 100, 256),
        Instruction(OpCode.ST, 502, 512, 0),
    ]
    
    print("Example instruction sequence for a single tile:\n")
    for i, inst in enumerate(examples, 1):
        encoded = inst.encode()
        print(f"{i}. {inst}")
        print(f"   Encoded: 0x{encoded:08X} ({bin(encoded)})\n")


def main():
    """Main demo runner"""
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == 'isa':
            show_isa_examples()
        elif mode == 'compare':
            demo_comparison()
        elif mode == 'quick':
            demo_512x512()
        else:
            print(f"Unknown mode: {mode}")
            print("Usage: python demo.py [quick|compare|isa]")
    else:
        # Default demo
        demo_512x512()


if __name__ == '__main__':
    main()
