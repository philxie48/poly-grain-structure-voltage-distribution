# Grain Growth and Electrical Conductivity Simulation Framework

A comprehensive multi-physics simulation framework for modeling grain growth and calculating electrical conductivity in polycrystalline platinum materials with quantum size effects.

## Overview

This framework implements a three-stage computational workflow:

1. **Phase Field Grain Growth Simulation** - Models polycrystalline microstructure evolution
2. **Grain Size Analysis and Preprocessing** - Extracts grain statistics and generates grain size distributions
3. **Finite Element Electrical Analysis** - Calculates potential distribution using the Mayadas-Shatzkes conductivity model

## Features

- **Multi-physics Coupling**: Seamless integration of microstructure and electrical simulations
- **Advanced Conductivity Model**: Mayadas-Shatzkes model with grain boundary scattering effects
- **High-Resolution Analysis**: 512×512 grid resolution for detailed microstructure characterization
- **Comprehensive Visualization**: Detailed plots of all computed fields and distributions
- **Material-Specific**: Optimized for platinum with quantum size effects (λ_mfp = 22.2 nm)

## Requirements

### Dependencies
```python
numpy>=1.20.0
matplotlib>=3.3.0
scipy>=1.7.0
tqdm>=4.62.0
```

### Installation
```bash
# Clone or download the repository
git clone <repository-url>
cd grain-growth-simulation

# Install dependencies
pip install numpy matplotlib scipy tqdm
```

## Workflow

### Step 1: Grain Growth Simulation

Generate polycrystalline microstructure using phase field method.

**Script:** `grain_growth_simulation.py`

```bash
python grain_growth_simulation.py
```

**Parameters:**
- Grid size: 512 × 512
- Number of orientations: 30
- Time steps: 5000
- Grid spacing: 2.0 µm
- Output interval: Every 100 steps

**Outputs:**
- `grain_growth_results/time_*.txt` - Order parameter φ distributions
- `grain_growth_results/*.jpg` - Microstructure visualizations

**Key Equations:**
```
∂η_k/∂t = -L[∂F/∂η_k - κ∇²η_k]
φ = Σᵢ ηᵢ²
```

### Step 2: Grain Size Analysis

Extract grain statistics and generate grain size distributions for conductivity calculations.

**Script:** `phi_to_dgrain_converter.py`

```bash
python phi_to_dgrain_converter.py
```

**Process:**
1. **Load φ distribution** from grain growth results
2. **Grain identification** using adaptive thresholding
3. **Connected component analysis** to identify individual grains
4. **Equivalent diameter calculation** based on grain area
5. **Pointwise grain size assignment** with φ-based corrections
6. **Data export** in compatible format for FEM analysis

**Outputs:**
- `*_dgrain.txt` - Pointwise grain size distribution
- `*_dgrain_analysis.png` - Grain analysis visualization

**Algorithm:**
```python
# Grain identification
grain_mask = phi > threshold
labeled_grains = connected_components(grain_mask)

# Equivalent diameter
d_equivalent = 2 * sqrt(area_pixels * dx² / π)

# Pointwise assignment
d_grain(i,j) = d_local * (φ(i,j)/φ_avg)^(1/3)
```

### Step 3: Electrical Conductivity Analysis

Calculate electrical potential distribution using finite element method with Mayadas-Shatzkes conductivity model.

**Script:** `fem_potential_solver.py`

```bash
python fem_potential_solver.py
```

**Process:**
1. **Load data** (φ and d_grain distributions)
2. **Calculate MS conductivity** at each grid point
3. **Assemble FEM system** using 4-node rectangular elements
4. **Apply boundary conditions** (5V left, 0V right, insulated top/bottom)
5. **Solve linear system** for potential distribution
6. **Post-process** electric field, current density, power loss
7. **Calculate effective properties** (σ_eff, resistance enhancement)

**Outputs:**
- `*_fem_results.png` - Comprehensive results visualization (9 subplots)
- Console output with effective conductivity and other metrics

**Key Models:**

**Mayadas-Shatzkes Conductivity:**
```
σ(φ) = σ_bulk × [1 - 3α/(2(1+α))]
α = λ_mfp × R / [d_grain × (1-R)]
R(φ) = R₀ × (1-φ)²
```

**Governing Equation:**
```
∇ · (σ(φ) ∇V) = 0
```

## Material Parameters

### Platinum Properties
- **Bulk conductivity**: σ_bulk = 9.43 × 10⁶ S/m
- **Mean free path**: λ_mfp = 22.2 × 10⁻⁹ m
- **Grain boundary reflection**: R₀ = 0.4
- **Grid spacing**: dx = 2.0 × 10⁻⁶ m

### Simulation Settings
- **Domain size**: 512 × 512 elements
- **Boundary conditions**: 5V (left) → 0V (right)
- **Element type**: 4-node bilinear quadrilateral
- **Solver**: Sparse direct solver (UMFPACK)

## Example Usage

### Complete Workflow
```bash
# Step 1: Generate microstructure
python grain_growth_simulation.py

# Step 2: Analyze grain sizes
python phi_to_dgrain_converter.py

# Step 3: Calculate electrical properties
python fem_potential_solver.py
```

### Custom Parameters
```python
# Modify phi_to_dgrain_converter.py for different files
phi_file = "grain_growth_results/time_1000.txt"
dx = 2.0e-6  # Grid spacing in meters

# Modify fem_potential_solver.py for different boundary conditions
V_left = 10.0   # Left boundary voltage
V_right = 0.0   # Right boundary voltage
```

## Output Interpretation

### Grain Growth Results
- **φ values**: 0 (grain boundary) → 1 (grain interior)
- **Microstructure**: Colored regions represent different grain orientations
- **Evolution**: Grain coarsening over time steps

### Grain Size Analysis
- **d_grain values**: Effective grain diameter in meters
- **Distribution**: Histogram shows grain size statistics
- **Labeled grains**: Color-coded individual grains

### Electrical Analysis
The final visualization includes 9 subplots:

1. **φ Distribution** - Original order parameter
2. **d_grain Distribution** - Grain size field (nm)
3. **Conductivity** - log₁₀(σ) distribution
4. **Potential** - Voltage distribution (V)
5. **Electric Field** - |E| magnitude (V/m)
6. **Current Density** - |J| magnitude (A/m²)
7. **Power Loss** - Energy dissipation (W/m³)
8. **Scattering Parameter** - α values
9. **Reflection Coefficient** - R values

### Key Metrics
```
Effective Conductivity: σ_eff [S/m]
Conductivity Ratio: σ_eff/σ_bulk
Resistance Enhancement: ρ_eff/ρ_bulk
Total Current: I_total [A]
Total Power: P_total [W]
```

## Performance Notes

### Computational Requirements
- **Memory**: ~2-4 GB RAM for 512² grids
- **Time**: 
  - Grain growth: ~10-30 minutes
  - Grain analysis: ~1-2 minutes  
  - FEM solution: ~15-30 seconds
- **Storage**: ~100 MB for complete workflow results

### Optimization Tips
- Use vectorized operations for large arrays
- Employ sparse matrices for FEM assembly
- Consider parallel processing for multiple time steps
- Monitor memory usage for very large grids

## Troubleshooting

### Common Issues

**1. File Not Found Errors**
```bash
# Ensure correct file paths
ls grain_growth_results/
# Check that time_*.txt files exist
```

**2. Memory Issues**
```python
# Reduce grid size for testing
Lx, Ly = 256, 256  # Instead of 512, 512
```

**3. Convergence Problems**
```python
# Check conductivity bounds
print(f"σ range: {np.min(sigma_grid):.2e} - {np.max(sigma_grid):.2e}")
# Ensure reasonable α values
print(f"α range: {np.min(alpha_grid):.2f} - {np.max(alpha_grid):.2f}")
```

**4. Visualization Issues**
```python
# Check data orientation
print(f"φ shape: {phi_grid.shape}")
print(f"V shape: {V_grid.shape}")
# Ensure consistent coordinate systems
```

### Validation Checks

**Physical Reasonableness:**
- σ_eff < σ_bulk (conductivity reduction due to grain boundaries)
- 0 ≤ φ ≤ 1 (order parameter bounds)
- Smooth potential distribution from left to right
- Current conservation: ∇ · J = 0

**Numerical Stability:**
- Well-conditioned FEM matrix
- Reasonable field magnitudes
- Smooth spatial distributions

## File Structure
```
project/
├── grain_growth_simulation.py      # Phase field grain growth
├── phi_to_dgrain_converter.py      # Grain size analysis
├── fem_potential_solver.py         # Electrical FEM solver
├── workflow.md                     # Detailed mathematical workflow
├── Theory_background.md            # Theoretical foundation
├── README.md                       # This file
└── grain_growth_results/           # Output directory
    ├── time_*.txt                  # φ distributions
    ├── time_*_dgrain.txt          # Grain size distributions
    ├── *_dgrain_analysis.png      # Grain analysis plots
    ├── *_fem_results.png          # FEM results plots
    └── *.jpg                       # Microstructure images
```

## Contributing

### Development Guidelines
- Follow PEP 8 style conventions
- Add comprehensive docstrings
- Include unit tests for new functions
- Validate against analytical solutions when possible

### Adding New Features
- Extend material database for other metals
- Implement adaptive mesh refinement
- Add parallel processing capabilities
- Include uncertainty quantification

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Mayadas-Shatzkes model implementation based on Physical Review B 1, 1382 (1970)
- Phase field method follows Chen L.-Q., Annual Review of Materials Research 32, 113 (2002)
- FEM implementation uses standard Galerkin formulation

## Contact

For questions, issues, or contributions, please contact [your-email@example.com] or open an issue in the repository.

---

**Version**: 1.0  
**Last Updated**: 2024  
**Compatibility**: Python 3.7+
