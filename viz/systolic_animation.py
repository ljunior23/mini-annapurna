import json
from typing import List, Tuple


def generate_animation_html(
    array_size: int = 8,  # Use smaller size for visualization
    tile_m: int = 8,
    tile_k: int = 8,
    tile_n: int = 8,
    output_file: str = "systolic_animation.html"
) -> None:
        
    # Generate data flow sequence
    cycles = []
    max_cycles = tile_k + array_size + 2  # Latency + compute + drain
    
    for cycle in range(max_cycles):
        cycle_data = {
            'cycle': cycle,
            'a_positions': [],  # A matrix elements (flow right)
            'b_positions': [],  # B matrix elements (flow down)
            'c_values': [],     # Accumulated C values
            'active_pes': []    # Which PEs are active
        }
        
        # Simulate data flow
        for i in range(array_size):
            for j in range(array_size):
                # Check if this PE is active this cycle
                # PE[i,j] starts at cycle (i+j) due to wavefront
                start_cycle = i + j
                end_cycle = start_cycle + tile_k
                
                if start_cycle <= cycle < end_cycle:
                    k = cycle - start_cycle
                    cycle_data['active_pes'].append({'i': i, 'j': j, 'k': k})
                    
                    # A flows horizontally
                    if i < tile_m and k < tile_k:
                        cycle_data['a_positions'].append({
                            'i': i, 'j': j, 'k': k, 'value': f'A[{i},{k}]'
                        })
                    
                    # B flows vertically
                    if j < tile_n and k < tile_k:
                        cycle_data['b_positions'].append({
                            'i': i, 'j': j, 'k': k, 'value': f'B[{k},{j}]'
                        })
                    
                    # C accumulates
                    if i < tile_m and j < tile_n:
                        cycle_data['c_values'].append({
                            'i': i, 'j': j, 'progress': min(100, (k+1) * 100 // tile_k)
                        })
        
        cycles.append(cycle_data)
    
    # Generate HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Systolic Array Animation - Mini-Annapurna</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            color: #333;
        }}
        
        h1 {{
            text-align: center;
            color: #667eea;
            margin-bottom: 10px;
        }}
        
        .subtitle {{
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }}
        
        .grid-container {{
            display: flex;
            justify-content: center;
            margin: 30px 0;
        }}
        
        .systolic-grid {{
            display: grid;
            grid-template-columns: repeat({array_size}, 60px);
            grid-template-rows: repeat({array_size}, 60px);
            gap: 4px;
            background: #f0f0f0;
            padding: 10px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        
        .pe {{
            width: 60px;
            height: 60px;
            background: white;
            border: 2px solid #ddd;
            border-radius: 8px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            font-size: 10px;
            transition: all 0.3s ease;
            position: relative;
        }}
        
        .pe.active {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-color: #667eea;
            color: white;
            transform: scale(1.1);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.5);
        }}
        
        .pe.computing {{
            background: #4CAF50;
            border-color: #45a049;
            color: white;
        }}
        
        .pe-label {{
            font-weight: bold;
            font-size: 9px;
        }}
        
        .pe-value {{
            font-size: 8px;
            margin-top: 2px;
        }}
        
        .progress-bar {{
            width: 50px;
            height: 4px;
            background: #ddd;
            border-radius: 2px;
            margin-top: 4px;
            overflow: hidden;
        }}
        
        .progress-fill {{
            height: 100%;
            background: #4CAF50;
            transition: width 0.3s ease;
        }}
        
        .controls {{
            text-align: center;
            margin: 20px 0;
        }}
        
        button {{
            background: #667eea;
            color: white;
            border: none;
            padding: 12px 24px;
            margin: 0 5px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            font-weight: bold;
            transition: all 0.3s ease;
        }}
        
        button:hover {{
            background: #5568d3;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }}
        
        button:disabled {{
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }}
        
        .info-panel {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin: 20px 0;
        }}
        
        .info-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        
        .info-card h3 {{
            margin: 0 0 10px 0;
            font-size: 14px;
            opacity: 0.9;
        }}
        
        .info-card .value {{
            font-size: 32px;
            font-weight: bold;
            margin: 0;
        }}
        
        .legend {{
            display: flex;
            justify-content: center;
            gap: 30px;
            margin: 20px 0;
            flex-wrap: wrap;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .legend-box {{
            width: 30px;
            height: 30px;
            border-radius: 5px;
            border: 2px solid #333;
        }}
        
        .legend-box.idle {{
            background: white;
            border-color: #ddd;
        }}
        
        .legend-box.active {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }}
        
        .legend-box.computing {{
            background: #4CAF50;
        }}
        
        .data-flow {{
            display: flex;
            justify-content: space-around;
            margin: 20px 0;
            padding: 20px;
            background: #f9f9f9;
            border-radius: 10px;
        }}
        
        .flow-column {{
            text-align: center;
        }}
        
        .flow-label {{
            font-weight: bold;
            color: #667eea;
            margin-bottom: 10px;
        }}
        
        .flow-arrow {{
            font-size: 40px;
            color: #667eea;
            margin: 10px 0;
        }}
        
        .speed-control {{
            text-align: center;
            margin: 20px 0;
        }}
        
        .speed-control input {{
            width: 200px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔲 Systolic Array Animation</h1>
        <div class="subtitle">Mini-Annapurna: {array_size}×{array_size} Array Computing {tile_m}×{tile_k}×{tile_n} Tile</div>
        
        <div class="info-panel">
            <div class="info-card">
                <h3>Current Cycle</h3>
                <div class="value" id="cycle-display">0</div>
            </div>
            <div class="info-card">
                <h3>Active PEs</h3>
                <div class="value" id="active-pes">0</div>
            </div>
            <div class="info-card">
                <h3>Utilization</h3>
                <div class="value" id="utilization">0%</div>
            </div>
        </div>
        
        <div class="legend">
            <div class="legend-item">
                <div class="legend-box idle"></div>
                <span>Idle</span>
            </div>
            <div class="legend-item">
                <div class="legend-box active"></div>
                <span>Active (Computing)</span>
            </div>
            <div class="legend-item">
                <div class="legend-box computing"></div>
                <span>Accumulating</span>
            </div>
        </div>
        
        <div class="grid-container">
            <div class="systolic-grid" id="systolic-grid">
                <!-- PEs will be generated here -->
            </div>
        </div>
        
        <div class="data-flow">
            <div class="flow-column">
                <div class="flow-label">Matrix A (Input)</div>
                <div class="flow-arrow">→</div>
                <div>Flows Right</div>
            </div>
            <div class="flow-column">
                <div class="flow-label">Matrix B (Weight)</div>
                <div class="flow-arrow">↓</div>
                <div>Flows Down</div>
            </div>
            <div class="flow-column">
                <div class="flow-label">Matrix C (Output)</div>
                <div class="flow-arrow">⊕</div>
                <div>Accumulates In-Place</div>
            </div>
        </div>
        
        <div class="controls">
            <button id="play-btn" onclick="togglePlay()">▶ Play</button>
            <button onclick="stepForward()">⏭ Step</button>
            <button onclick="reset()">⏮ Reset</button>
        </div>
        
        <div class="speed-control">
            <label>Animation Speed: </label>
            <input type="range" id="speed-slider" min="100" max="2000" value="500" step="100">
            <span id="speed-label">500ms</span>
        </div>
        
        <div style="margin-top: 30px; padding: 20px; background: #f9f9f9; border-radius: 10px;">
            <h3 style="color: #667eea;">How It Works</h3>
            <p><strong>Systolic arrays</strong> process matrix multiplication by flowing data through a grid of processing elements (PEs).</p>
            <ul>
                <li><strong>A matrix</strong> elements flow horizontally (left to right)</li>
                <li><strong>B matrix</strong> elements flow vertically (top to bottom)</li>
                <li>Each PE computes: <code>C += A × B</code></li>
                <li>Data is reused as it flows through multiple PEs</li>
                <li>Wavefront execution: PEs start computing at staggered times</li>
            </ul>
            <p><strong>Key Insight:</strong> Each data element is used {array_size} times as it flows through the array, minimizing memory bandwidth!</p>
        </div>
    </div>

    <script>
        const cycles = {json.dumps(cycles, indent=2)};
        const arraySize = {array_size};
        let currentCycle = 0;
        let isPlaying = false;
        let animationInterval = null;
        let animationSpeed = 500;
        
        // Initialize grid
        function initGrid() {{
            const grid = document.getElementById('systolic-grid');
            grid.innerHTML = '';
            
            for (let i = 0; i < arraySize; i++) {{
                for (let j = 0; j < arraySize; j++) {{
                    const pe = document.createElement('div');
                    pe.className = 'pe';
                    pe.id = `pe-${{i}}-${{j}}`;
                    pe.innerHTML = `
                        <div class="pe-label">PE[${{i}},${{j}}]</div>
                        <div class="pe-value" id="val-${{i}}-${{j}}"></div>
                        <div class="progress-bar">
                            <div class="progress-fill" id="prog-${{i}}-${{j}}" style="width: 0%"></div>
                        </div>
                    `;
                    grid.appendChild(pe);
                }}
            }}
        }}
        
        // Update display for current cycle
        function updateDisplay() {{
            if (currentCycle >= cycles.length) {{
                pause();
                return;
            }}
            
            const cycleData = cycles[currentCycle];
            
            // Update cycle info
            document.getElementById('cycle-display').textContent = cycleData.cycle;
            document.getElementById('active-pes').textContent = cycleData.active_pes.length;
            const util = Math.round((cycleData.active_pes.length / (arraySize * arraySize)) * 100);
            document.getElementById('utilization').textContent = util + '%';
            
            // Reset all PEs
            for (let i = 0; i < arraySize; i++) {{
                for (let j = 0; j < arraySize; j++) {{
                    const pe = document.getElementById(`pe-${{i}}-${{j}}`);
                    pe.className = 'pe';
                    document.getElementById(`val-${{i}}-${{j}}`).textContent = '';
                }}
            }}
            
            // Highlight active PEs
            cycleData.active_pes.forEach(pe => {{
                const peElem = document.getElementById(`pe-${{pe.i}}-${{pe.j}}`);
                peElem.className = 'pe active';
                document.getElementById(`val-${{pe.i}}-${{pe.j}}`).textContent = `k=${{pe.k}}`;
            }});
            
            // Update C progress
            cycleData.c_values.forEach(c => {{
                const prog = document.getElementById(`prog-${{c.i}}-${{c.j}}`);
                if (prog) {{
                    prog.style.width = c.progress + '%';
                    if (c.progress >= 100) {{
                        document.getElementById(`pe-${{c.i}}-${{c.j}}`).classList.add('computing');
                    }}
                }}
            }});
        }}
        
        function stepForward() {{
            if (currentCycle < cycles.length - 1) {{
                currentCycle++;
                updateDisplay();
            }}
        }}
        
        function reset() {{
            currentCycle = 0;
            updateDisplay();
        }}
        
        function play() {{
            isPlaying = true;
            document.getElementById('play-btn').textContent = '⏸ Pause';
            animationInterval = setInterval(() => {{
                stepForward();
                if (currentCycle >= cycles.length - 1) {{
                    pause();
                }}
            }}, animationSpeed);
        }}
        
        function pause() {{
            isPlaying = false;
            document.getElementById('play-btn').textContent = '▶ Play';
            if (animationInterval) {{
                clearInterval(animationInterval);
                animationInterval = null;
            }}
        }}
        
        function togglePlay() {{
            if (isPlaying) {{
                pause();
            }} else {{
                play();
            }}
        }}
        
        // Speed control
        document.getElementById('speed-slider').addEventListener('input', (e) => {{
            animationSpeed = parseInt(e.target.value);
            document.getElementById('speed-label').textContent = animationSpeed + 'ms';
            if (isPlaying) {{
                pause();
                play();
            }}
        }});
        
        // Initialize
        initGrid();
        updateDisplay();
    </script>
</body>
</html>
"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"Systolic array animation saved to {output_file}")
    print(f"   Open in browser to see the visualization!")


if __name__ == "__main__":
    import sys
    
    output = sys.argv[1] if len(sys.argv) > 1 else "systolic_animation.html"
    generate_animation_html(output_file=output)
