import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict


def load_trace(filename: str) -> List[Dict]:
    """Load trace from CSV file"""
    trace = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            trace.append(row)
    return trace


def analyze_trace(trace: List[Dict]) -> Dict:
    """
    Analyze trace and compute metrics for roofline plot.
    """
    mac_ops = sum(1 for row in trace if row['opcode'] == 'MAC')
    load_ops = sum(1 for row in trace if row['opcode'] == 'LD')
    store_ops = sum(1 for row in trace if row['opcode'] == 'ST')
    
    # Estimate metrics (simplified)
    # Assume 256×256×256 tile for MACs
    flops_per_mac = 2 * 256 * 256 * 256  # 2 FLOPs per MAC, MxKxN tile
    total_flops = mac_ops * flops_per_mac
    
    # Bytes per tile (256×256 × 4 bytes)
    bytes_per_tile = 256 * 256 * 4
    total_bytes = (load_ops + store_ops) * bytes_per_tile
    
    # Operational intensity (FLOPs/byte)
    operational_intensity = total_flops / total_bytes if total_bytes > 0 else 0
    
    # Get total cycles from last trace entry
    total_cycles = int(trace[-1]['cycle']) if trace else 0
    
    # Assume 1 GHz clock
    performance_gflops = (total_flops / total_cycles) if total_cycles > 0 else 0
    
    return {
        'mac_ops': mac_ops,
        'total_flops': total_flops,
        'total_bytes': total_bytes,
        'operational_intensity': operational_intensity,
        'total_cycles': total_cycles,
        'performance_gflops': performance_gflops
    }


def plot_roofline(trace_file: str, output_file: str = 'roofline.png'):
    """
    Generate roofline plot from trace.
    """
    print(f"Loading trace from {trace_file}...")
    trace = load_trace(trace_file)
    
    print(f"Analyzing {len(trace)} trace entries...")
    metrics = analyze_trace(trace)
    
    print(f"\nRoofline Metrics:")
    print(f"  Total FLOPs:              {metrics['total_flops']:,}")
    print(f"  Total Bytes:              {metrics['total_bytes']:,}")
    print(f"  Operational Intensity:    {metrics['operational_intensity']:.2f} FLOPs/byte")
    print(f"  Performance:              {metrics['performance_gflops']:.2f} GFLOP/s")
    
    # Create roofline plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # System parameters
    peak_gflops = 256 * 256 * 2 * 1.0  # 256×256 array, 2 FLOPs/MAC, 1 GHz
    memory_bw_gbs = 100.0  # 100 GB/s
    
    # Operational intensity range
    oi_range = np.logspace(-1, 2, 100)
    
    # Memory-bound region (performance = BW × OI)
    memory_bound = memory_bw_gbs * oi_range
    
    # Compute-bound region (performance = peak)
    compute_bound = np.ones_like(oi_range) * peak_gflops
    
    # Roofline (minimum of both)
    roofline = np.minimum(memory_bound, compute_bound)
    
    # Plot roofline
    ax.loglog(oi_range, roofline, 'k-', linewidth=2, label='Roofline')
    ax.loglog(oi_range, memory_bound, 'b--', alpha=0.5, label='Memory Bound')
    ax.axhline(peak_gflops, color='r', linestyle='--', alpha=0.5, label='Compute Bound')
    
    # Plot actual performance
    if metrics['operational_intensity'] > 0:
        ax.loglog(metrics['operational_intensity'], 
                 metrics['performance_gflops'],
                 'go', markersize=12, label='Actual Performance')
    
    # Formatting
    ax.set_xlabel('Operational Intensity (FLOPs/byte)', fontsize=12)
    ax.set_ylabel('Performance (GFLOP/s)', fontsize=12)
    ax.set_title('Mini-Annapurna Roofline Model', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Add annotations
    ax.text(0.5, peak_gflops * 0.6, 
            f'Peak: {peak_gflops:.0f} GFLOP/s\nBW: {memory_bw_gbs:.0f} GB/s',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nRoofline plot saved to {output_file}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Generate roofline plot')
    parser.add_argument('trace_file', type=str, help='Input trace CSV file')
    parser.add_argument('-o', '--output', type=str, default='roofline.png',
                        help='Output plot file')
    
    args = parser.parse_args()
    
    plot_roofline(args.trace_file, args.output)


if __name__ == '__main__':
    main()
