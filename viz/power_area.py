
from dataclasses import dataclass
from typing import Dict


@dataclass
class TechnologyNode:
    """Process technology parameters."""
    name: str
    node_nm: int  # Process node in nm
    voltage: float  # Operating voltage in V
    freq_ghz: float  # Typical frequency in GHz
    
    # Power coefficients (calibrated from literature)
    mac_power_mw: float  # Power per MAC per GHz
    sram_power_nj_per_bit: float  # SRAM access energy
    dram_power_nj_per_bit: float  # DRAM access energy
    
    # Area coefficients
    mac_area_mm2: float  # Area per MAC unit
    sram_area_mm2_per_kb: float  # SRAM area
    
    # Leakage power
    leakage_factor: float  # Leakage as % of dynamic


# Common technology nodes
TECH_NODES = {
    '28nm': TechnologyNode(
        name='28nm (TPU v1)',
        node_nm=28,
        voltage=0.9,
        freq_ghz=0.7,
        mac_power_mw=0.3,
        sram_power_nj_per_bit=1.5,
        dram_power_nj_per_bit=50.0,
        mac_area_mm2=0.01,
        sram_area_mm2_per_kb=0.05,
        leakage_factor=0.15
    ),
    '16nm': TechnologyNode(
        name='16nm (TPU v2)',
        node_nm=16,
        voltage=0.8,
        freq_ghz=1.0,
        mac_power_mw=0.15,
        sram_power_nj_per_bit=0.8,
        dram_power_nj_per_bit=35.0,
        mac_area_mm2=0.006,
        sram_area_mm2_per_kb=0.03,
        leakage_factor=0.20
    ),
    '7nm': TechnologyNode(
        name='7nm (TPU v4, Modern)',
        node_nm=7,
        voltage=0.75,
        freq_ghz=1.4,
        mac_power_mw=0.08,
        sram_power_nj_per_bit=0.4,
        dram_power_nj_per_bit=25.0,
        mac_area_mm2=0.003,
        sram_area_mm2_per_kb=0.015,
        leakage_factor=0.25
    ),
    '5nm': TechnologyNode(
        name='5nm (Cutting Edge)',
        node_nm=5,
        voltage=0.7,
        freq_ghz=2.0,
        mac_power_mw=0.05,
        sram_power_nj_per_bit=0.25,
        dram_power_nj_per_bit=20.0,
        mac_area_mm2=0.002,
        sram_area_mm2_per_kb=0.01,
        leakage_factor=0.30
    )
}


@dataclass
class PowerAreaEstimate:
    """Power and area estimation results."""
    # Area estimates
    mac_array_area_mm2: float
    scratchpad_area_mm2: float
    control_area_mm2: float
    total_area_mm2: float
    
    # Power estimates (per inference)
    compute_energy_mj: float  # mJ = millijoules
    sram_energy_mj: float
    dram_energy_mj: float
    leakage_energy_mj: float
    total_energy_mj: float
    
    # Power (at given frequency)
    compute_power_w: float
    memory_power_w: float
    leakage_power_w: float
    total_power_w: float
    
    # Efficiency metrics
    tops: float  # Tera-ops per second
    tops_per_watt: float
    energy_per_mac_pj: float  # pJ = picojoules
    
    # Comparison to real accelerators
    vs_tpu_v1: Dict[str, float]
    vs_a100: Dict[str, float]


def estimate_power_area(
    array_rows: int = 256,
    array_cols: int = 256,
    scratchpad_kb: int = 1024,
    total_macs: int = 134_217_728,  # For 512x512 matmul
    total_cycles: int = 65_704,
    bytes_transferred: int = 6_291_456,
    tech_node: str = '7nm'
) -> PowerAreaEstimate:
   
    tech = TECH_NODES[tech_node]
    
    
    # MAC array area
    num_macs = array_rows * array_cols
    mac_array_area_mm2 = num_macs * tech.mac_area_mm2
    
    # Scratchpad area (assume 32-bit = 4 bytes)
    scratchpad_area_mm2 = scratchpad_kb * tech.sram_area_mm2_per_kb
    
    # Control logic area (estimated as 20% of compute)
    control_area_mm2 = mac_array_area_mm2 * 0.2
    
    # Total area
    total_area_mm2 = mac_array_area_mm2 + scratchpad_area_mm2 + control_area_mm2
    
       
    # Time for workload (in seconds)
    # freq_ghz is already in GHz, cycles are at that frequency
    time_seconds = total_cycles / (tech.freq_ghz * 1e9)  # Convert GHz to Hz
    
    # Compute energy (dynamic power)
    # Each MAC: multiply (1 op) + accumulate (1 op) = 2 FLOPs
    compute_energy_j = (total_macs * tech.mac_power_mw * 1e-3) / tech.freq_ghz
    compute_energy_mj = compute_energy_j * 1000
    
    # SRAM energy
    # Assume each MAC reads 2 operands + writes 1 result = 3 SRAM accesses
    # Each access is 32 bits
    sram_accesses = total_macs * 3 * 32  # bits
    sram_energy_j = sram_accesses * tech.sram_power_nj_per_bit * 1e-9
    sram_energy_mj = sram_energy_j * 1000
    
    # DRAM energy
    dram_bits = bytes_transferred * 8
    dram_energy_j = dram_bits * tech.dram_power_nj_per_bit * 1e-9
    dram_energy_mj = dram_energy_j * 1000
    
    # Leakage energy
    dynamic_energy_j = compute_energy_j + sram_energy_j + dram_energy_j
    leakage_energy_j = dynamic_energy_j * tech.leakage_factor
    leakage_energy_mj = leakage_energy_j * 1000
    
    # Total energy
    total_energy_mj = compute_energy_mj + sram_energy_mj + dram_energy_mj + leakage_energy_mj
    
    # Average power
    compute_power_w = compute_energy_j / time_seconds
    memory_power_w = (sram_energy_j + dram_energy_j) / time_seconds
    leakage_power_w = leakage_energy_j / time_seconds
    total_power_w = total_energy_mj / (time_seconds * 1000)
    
    
    # TOPS (Tera-ops per second)
    flops = total_macs * 2  # 2 FLOPs per MAC
    tops = (flops / time_seconds) / 1e12
    
    # TOPS/Watt
    tops_per_watt = tops / total_power_w if total_power_w > 0 else 0
    
    # Energy per MAC (picojoules)
    energy_per_mac_pj = (total_energy_mj * 1e-3 / total_macs) * 1e12
    
    
    # TPU v1 estimates (from Google's paper)
    tpu_v1_tops = 92  # 92 TOPS for TPU v1
    tpu_v1_power = 40  # ~40W
    tpu_v1_area = 331  # 331 mm²
    
    vs_tpu_v1 = {
        'tops_ratio': tops / tpu_v1_tops,
        'power_ratio': total_power_w / tpu_v1_power,
        'area_ratio': total_area_mm2 / tpu_v1_area,
        'efficiency_ratio': tops_per_watt / (tpu_v1_tops / tpu_v1_power)
    }
    
    # NVIDIA A100 Tensor Core estimates
    a100_tops = 312  # 312 TOPS (FP16)
    a100_power = 400  # ~400W TDP
    a100_area = 826  # 826 mm²
    
    vs_a100 = {
        'tops_ratio': tops / a100_tops,
        'power_ratio': total_power_w / a100_power,
        'area_ratio': total_area_mm2 / a100_area,
        'efficiency_ratio': tops_per_watt / (a100_tops / a100_power)
    }
    
    return PowerAreaEstimate(
        mac_array_area_mm2=mac_array_area_mm2,
        scratchpad_area_mm2=scratchpad_area_mm2,
        control_area_mm2=control_area_mm2,
        total_area_mm2=total_area_mm2,
        compute_energy_mj=compute_energy_mj,
        sram_energy_mj=sram_energy_mj,
        dram_energy_mj=dram_energy_mj,
        leakage_energy_mj=leakage_energy_mj,
        total_energy_mj=total_energy_mj,
        compute_power_w=compute_power_w,
        memory_power_w=memory_power_w,
        leakage_power_w=leakage_power_w,
        total_power_w=total_power_w,
        tops=tops,
        tops_per_watt=tops_per_watt,
        energy_per_mac_pj=energy_per_mac_pj,
        vs_tpu_v1=vs_tpu_v1,
        vs_a100=vs_a100
    )


def print_power_area_report(estimate: PowerAreaEstimate, tech_node: str = '7nm'):
    """Print a formatted power and area report."""
    tech = TECH_NODES[tech_node]
    
    print("\n" + "="*70)
    print(f"  POWER & AREA ESTIMATION - Mini-Annapurna @ {tech.name}")
    print("="*70)
    
    print("\n AREA BREAKDOWN")
    print("-" * 70)
    print(f"  MAC Array:        {estimate.mac_array_area_mm2:8.2f} mm²  "
          f"({estimate.mac_array_area_mm2/estimate.total_area_mm2*100:.1f}%)")
    print(f"  Scratchpad SRAM:  {estimate.scratchpad_area_mm2:8.2f} mm²  "
          f"({estimate.scratchpad_area_mm2/estimate.total_area_mm2*100:.1f}%)")
    print(f"  Control Logic:    {estimate.control_area_mm2:8.2f} mm²  "
          f"({estimate.control_area_mm2/estimate.total_area_mm2*100:.1f}%)")
    print(f"  {'─'*66}")
    print(f"  TOTAL CHIP AREA:  {estimate.total_area_mm2:8.2f} mm²")
    
    print("\n ENERGY BREAKDOWN (per inference)")
    print("-" * 70)
    print(f"  Compute (MACs):   {estimate.compute_energy_mj:8.3f} mJ  "
          f"({estimate.compute_energy_mj/estimate.total_energy_mj*100:.1f}%)")
    print(f"  SRAM Access:      {estimate.sram_energy_mj:8.3f} mJ  "
          f"({estimate.sram_energy_mj/estimate.total_energy_mj*100:.1f}%)")
    print(f"  DRAM Access:      {estimate.dram_energy_mj:8.3f} mJ  "
          f"({estimate.dram_energy_mj/estimate.total_energy_mj*100:.1f}%)")
    print(f"  Leakage:          {estimate.leakage_energy_mj:8.3f} mJ  "
          f"({estimate.leakage_energy_mj/estimate.total_energy_mj*100:.1f}%)")
    print(f"  {'─'*66}")
    print(f"  TOTAL ENERGY:     {estimate.total_energy_mj:8.3f} mJ")
    
    print("\n POWER BREAKDOWN")
    print("-" * 70)
    print(f"  Compute Power:    {estimate.compute_power_w:8.2f} W")
    print(f"  Memory Power:     {estimate.memory_power_w:8.2f} W")
    print(f"  Leakage Power:    {estimate.leakage_power_w:8.2f} W")
    print(f"  {'─'*66}")
    print(f"  TOTAL POWER:      {estimate.total_power_w:8.2f} W")
    
    print("\n EFFICIENCY METRICS")
    print("-" * 70)
    print(f"  Peak Performance: {estimate.tops:8.2f} TOPS")
    print(f"  Power Efficiency: {estimate.tops_per_watt:8.2f} TOPS/Watt")
    print(f"  Energy per MAC:   {estimate.energy_per_mac_pj:8.2f} pJ/MAC")
    
    print("\n COMPARISON TO REAL ACCELERATORS")
    print("-" * 70)
    print("  vs Google TPU v1:")
    print(f"    Performance:    {estimate.vs_tpu_v1['tops_ratio']*100:6.1f}% "
          f"({estimate.tops:.1f} vs 92 TOPS)")
    print(f"    Power:          {estimate.vs_tpu_v1['power_ratio']*100:6.1f}% "
          f"({estimate.total_power_w:.1f}W vs 40W)")
    print(f"    Area:           {estimate.vs_tpu_v1['area_ratio']*100:6.1f}% "
          f"({estimate.total_area_mm2:.1f} vs 331 mm²)")
    print(f"    Efficiency:     {estimate.vs_tpu_v1['efficiency_ratio']*100:6.1f}% "
          f"of TPU v1")
    
    print("\n  vs NVIDIA A100 Tensor Core:")
    print(f"    Performance:    {estimate.vs_a100['tops_ratio']*100:6.1f}% "
          f"({estimate.tops:.1f} vs 312 TOPS)")
    print(f"    Power:          {estimate.vs_a100['power_ratio']*100:6.1f}% "
          f"({estimate.total_power_w:.1f}W vs 400W)")
    print(f"    Area:           {estimate.vs_a100['area_ratio']*100:6.1f}% "
          f"({estimate.total_area_mm2:.1f} vs 826 mm²)")
    print(f"    Efficiency:     {estimate.vs_a100['efficiency_ratio']*100:6.1f}% "
          f"of A100")
    
    print("\n KEY INSIGHTS")
    print("-" * 70)
    
    # Analyze power breakdown
    mem_pct = (estimate.memory_power_w / estimate.total_power_w) * 100
    if mem_pct > 50:
        print(f"   Memory dominates power ({mem_pct:.0f}%) - bandwidth is bottleneck")
    else:
        print(f"  Compute dominates power ({100-mem_pct:.0f}%) - well-balanced")
    
    # Analyze area
    sram_pct = (estimate.scratchpad_area_mm2 / estimate.total_area_mm2) * 100
    if sram_pct > 50:
        print(f"  SRAM dominates area ({sram_pct:.0f}%) - typical for accelerators")
    
    # Compare efficiency
    if estimate.tops_per_watt > 2.0:
        print(f" Excellent efficiency: {estimate.tops_per_watt:.1f} TOPS/W")
    elif estimate.tops_per_watt > 1.0:
        print(f"  Good efficiency: {estimate.tops_per_watt:.1f} TOPS/W")
    else:
        print(f"  Low efficiency: {estimate.tops_per_watt:.1f} TOPS/W")
    
    print("\n" + "="*70 + "\n")


def generate_comparison_chart(tech_nodes: list = ['28nm', '16nm', '7nm', '5nm']):
    """Generate comparison across technology nodes."""
    print("\n" + "="*80)
    print("  TECHNOLOGY SCALING COMPARISON")
    print("="*80 + "\n")
    
    print(f"{'Tech Node':<12} {'Area (mm²)':<15} {'Power (W)':<12} {'TOPS':<10} {'TOPS/W':<10}")
    print("-" * 80)
    
    for node in tech_nodes:
        est = estimate_power_area(tech_node=node)
        print(f"{node:<12} {est.total_area_mm2:>10.1f}     {est.total_power_w:>8.1f}    "
              f"{est.tops:>6.1f}    {est.tops_per_watt:>6.2f}")
    
    print("\n")


if __name__ == "__main__":
   
    print("\n Power & Area Estimation for 512×512 MatMul")
    
    estimate = estimate_power_area(
        array_rows=256,
        array_cols=256,
        scratchpad_kb=1024,
        total_macs=134_217_728,
        total_cycles=65_704,
        bytes_transferred=6_291_456,
        tech_node='7nm'
    )
    
    print_power_area_report(estimate, '7nm')
    
    # Show scaling
    generate_comparison_chart()
