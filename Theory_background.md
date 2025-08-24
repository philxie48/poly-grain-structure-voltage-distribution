# Theory Background: Grain Growth and Electrical Conductivity Simulation

## Overview

This document provides the theoretical foundation for the multi-physics simulation framework that combines phase-field grain growth modeling with finite element electrical conductivity analysis. The framework is specifically designed for platinum (Pt) materials with quantum size effects and grain boundary scattering considerations.

## 1. Phase Field Method for Grain Growth

### 1.1 Fundamental Equations

The phase field method models grain growth using multiple order parameters η_k(x,y,t), where each η_k represents a different grain orientation. The evolution equations are:

```
∂η_k/∂t = -L * [∂F/∂η_k - κ ∇²η_k]
```

Where:
- `η_k`: Order parameter for grain orientation k
- `L`: Kinetic coefficient
- `F`: Free energy functional
- `κ`: Gradient energy coefficient
- `∇²`: Laplacian operator

### 1.2 Free Energy Functional

The total free energy includes chemical and gradient contributions:

```
F = ∫[f(η₁,η₂,...,η_P) + κ/2 ∑ᵢ|∇ηᵢ|²] dV
```

The chemical free energy density is:
```
f(η₁,...,η_P) = α∑ᵢηᵢ² - β∑ᵢηᵢ⁴ + γ∑ᵢ∑ⱼ≠ᵢηᵢ²ηⱼ²
```

Where:
- `α, β, γ`: Model parameters controlling grain boundary energy
- `P`: Number of grain orientations (typically 30)

### 1.3 Order Parameter Calculation

The local grain structure is characterized by the total order parameter:
```
φ(x,y) = ∑ᵢ₌₁ᴾ ηᵢ²(x,y)
```

Where:
- `φ = 1`: Perfect grain interior
- `φ ≈ 0`: Grain boundary region
- `0 < φ < 1`: Transition zone

## 2. Electrical Conductivity Theory

### 2.1 Governing Equation

The electrical potential distribution is governed by the steady-state current conservation equation:
```
∇ · (σ(φ) ∇V) = 0
```

Where:
- `V(x,y)`: Electrical potential [V]
- `σ(φ)`: Conductivity as a function of grain structure [S/m]
- `φ(x,y)`: Order parameter from phase field simulation

### 2.2 Boundary Conditions

**Dirichlet Conditions:**
- Left boundary: `V = V_left = 5.0 V`
- Right boundary: `V = V_right = 0.0 V`

**Neumann Conditions:**
- Top/bottom boundaries: `∂V/∂n = 0` (insulated)

### 2.3 Post-Processing Calculations

**Electric Field:**
```
E⃗ = -∇V = (-∂V/∂x, -∂V/∂y)
```

**Current Density:**
```
J⃗ = σ(φ) E⃗
```

**Power Dissipation:**
```
P = J⃗ · E⃗ = σ(φ)|E⃗|²
```

**Total Current:**
```
I_total = ∫_{right boundary} J_x dy
```

**Effective Conductivity:**
```
σ_eff = I_total × L_sample / (V_applied × A_cross)
```

## 3. Mayadas-Shatzkes Conductivity Model

### 3.1 Physical Foundation

The Mayadas-Shatzkes (MS) model describes electrical conductivity degradation due to electron scattering at grain boundaries. It accounts for:

1. **Size Effects**: When grain size approaches the mean free path
2. **Grain Boundary Scattering**: Reflection of electrons at interfaces
3. **Quantum Confinement**: Enhanced scattering in nanoscale structures

### 3.2 MS Conductivity Formula

```
σ(φ) = σ_bulk × [1 - (3/2) × α(φ)/(1 + α(φ))]
```

Where the scattering parameter is:
```
α(φ) = λ_mfp × R(φ) / [d_grain(φ) × (1 - R(φ))]
```

### 3.3 Model Parameters

**Material Constants (Platinum):**
- `σ_bulk = 9.43 × 10⁶ S/m`: Bulk conductivity
- `λ_mfp = 22.2 × 10⁻⁹ m`: Electron mean free path
- `R₀ = 0.4`: Reference grain boundary reflection coefficient

**Reflection Coefficient Model:**
```
R(φ) = R₀ × (1 - φ)²
```

Physical interpretation:
- `φ = 1`: `R = 0` (perfect grain, no scattering)
- `φ = 0`: `R = R₀` (pure grain boundary, maximum scattering)

**Effective Grain Size Model:**
```
d_grain(φ) = d_local × φ^(1/3)
```

Where `d_local` is determined through grain analysis of the φ distribution.

### 3.4 Multi-Scale Parameter Calculation

The MS model requires parameters at different length scales:

1. **Global Scale**: Material properties (σ_bulk, λ_mfp, R₀)
2. **Local Scale**: Grain-specific properties (d_local for each grain)
3. **Point Scale**: Pointwise values (φ, R(φ), α(φ), σ(φ))

## 4. Grain Size Analysis Algorithm

### 4.1 Grain Identification

**Thresholding:**
```
Grain mask = φ(x,y) > φ_threshold
```
Recommended: `φ_threshold = min(0.8, 75th percentile of φ)`

**Connected Component Analysis:**
- Label connected regions using 8-connectivity
- Filter out small regions (< 10 pixels) as noise

### 4.2 Equivalent Diameter Calculation

For each identified grain:
```
A_grain = N_pixels × (Δx)²
d_equivalent = 2 × √(A_grain / π)
```

Where:
- `N_pixels`: Number of pixels in the grain
- `Δx`: Grid spacing (2.0 × 10⁻⁶ m)

### 4.3 Pointwise Grain Size Assignment

```
d_grain(i,j) = {
  d_local × φ_correction,  if inside identified grain
  d_global × φ^(1/3),      if in transition region
  d_global × 0.1,          if in grain boundary region
}
```

Where:
- `φ_correction = (φ_local / φ_avg_grain)^(1/3)`
- `d_global`: Average grain size across entire domain

## 5. Finite Element Method Implementation

### 5.1 Weak Form

The weak formulation of the governing equation is:
```
∫_Ω σ(φ) ∇w · ∇V dΩ = 0  ∀w ∈ H₁₀(Ω)
```

Where `w` is the test function.

### 5.2 Discretization

**Grid:** 512 × 512 rectangular elements
**Nodes:** 513 × 513 nodes
**Element Type:** 4-node bilinear quadrilateral

**Shape Functions (for rectangular elements):**
```
N₁ = (1-ξ)(1-η)/4,  N₂ = (1+ξ)(1-η)/4
N₃ = (1+ξ)(1+η)/4,  N₄ = (1-ξ)(1+η)/4
```

### 5.3 Element Stiffness Matrix

```
K_e^{ij} = ∫_Ωe σ(φ) [∂Nᵢ/∂x ∂Nⱼ/∂x + ∂Nᵢ/∂y ∂Nⱼ/∂y] dΩ
```

For rectangular elements with constant conductivity:
```
K_e^{ij} = σ_element × [dNᵢ_dx × dNⱼ_dx + dNᵢ_dy × dNⱼ_dy] × Area
```

### 5.4 Assembly and Solution

1. **Global Assembly:** `K_global = ∑_elements K_element`
2. **Boundary Conditions:** Modify K and F for Dirichlet conditions
3. **Linear Solution:** `K V = F` using sparse direct solver
4. **Post-Processing:** Calculate E, J, and other derived quantities

## 6. Numerical Considerations

### 6.1 Stability and Convergence

- **CFL Condition**: `Δt ≤ (Δx)²/(2D)` for phase field evolution
- **Mesh Independence**: Grid spacing should resolve grain boundaries
- **MS Model Limits**: Apply bounds to prevent numerical instabilities

### 6.2 Computational Efficiency

- **Sparse Matrices**: Use CSR format for FEM assembly
- **Vectorization**: Optimize phase field calculations
- **Memory Management**: Efficient handling of large (512²) grids

## 7. Validation and Physical Interpretation

### 7.1 Expected Results

- **Conductivity Reduction**: σ_eff < σ_bulk due to grain boundary scattering
- **Resistance Enhancement**: ρ_eff/ρ_bulk > 1
- **Size Effects**: Stronger reduction for smaller grains

### 7.2 Key Output Metrics

- **Effective Conductivity**: σ_eff [S/m]
- **Resistance Enhancement Factor**: ρ_eff/ρ_bulk
- **Current Distribution**: Visualization of current paths
- **Power Loss**: Energy dissipation due to grain boundaries

## References

1. Mayadas, A. F., & Shatzkes, M. (1970). Electrical-resistivity model for polycrystalline films. Physical Review B, 1(4), 1382-1389.
2. Chen, L. Q. (2002). Phase-field models for microstructure evolution. Annual Review of Materials Research, 32(1), 113-140.
3. Moelans, N., Blanpain, B., & Wollants, P. (2008). An introduction to phase-field modeling of microstructure evolution. Calphad, 32(2), 268-294.
4. Zhang, L., Chen, L. Q., & Du, Q. (2008). Morphology of critical nuclei in solid-state phase transformations. Physical Review Letters, 98(26), 265703.
