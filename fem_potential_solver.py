import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
import time
import os

class FEMPotentialSolver:
    """
    FEM potential distribution solver based on MS conductivity model
    Uses 512×512 rectangular element grid
    """
    
    def __init__(self, Lx=512, Ly=512, dx=2.0e-6):
        """
        Initialize FEM solver
        
        Parameters:
        - Lx, Ly: Grid dimensions
        - dx: Grid spacing [m]
        """
        self.Lx = Lx
        self.Ly = Ly
        self.dx = dx
        self.dy = dx  # Assume square grid
        
        # Number of nodes and elements
        self.n_nodes_x = Lx + 1  # 513 nodes
        self.n_nodes_y = Ly + 1  # 513 nodes
        self.n_nodes = self.n_nodes_x * self.n_nodes_y  # Total nodes
        self.n_elements = Lx * Ly  # Number of elements
        
        print(f"FEM grid information:")
        print(f"Grid size: {Lx} × {Ly}")
        print(f"Number of nodes: {self.n_nodes_x} × {self.n_nodes_y} = {self.n_nodes}")
        print(f"Number of elements: {self.n_elements}")
        print(f"Grid spacing: {dx*1e6:.1f} μm")
        
        # MS model material parameters
        self.material_params = {
            'sigma_bulk': 9.43e6,      # S/m, bulk conductivity of platinum
            'lambda_mfp': 22.2e-9,     # m, electron mean free path
            'R0': 0.4,                 # grain boundary reflection coefficient
            'sigma_min': 9.43e-2,      # S/m, minimum conductivity
            'phi_min': 0.01,           # minimum φ value
            'R_max': 0.99,             # maximum reflection coefficient
            'alpha_max': 50.0          # maximum scattering intensity
        }
    
    def node_to_index(self, i, j):
        """
        Convert node coordinates (i,j) to global index
        i: X direction index (0 to Lx)
        j: Y direction index (0 to Ly)
        """
        return i * self.n_nodes_y + j
    
    def index_to_node(self, idx):
        """Convert global index to node coordinates"""
        i = idx // self.n_nodes_y
        j = idx % self.n_nodes_y
        return i, j
    
    def load_phi_and_dgrain_data(self, phi_file, dgrain_file):
        """
        Load φ and d_grain data
        
        Parameters:
        - phi_file: φ data file path
        - dgrain_file: d_grain data file path
        
        Returns:
        - phi_grid: 512×512 φ distribution
        - dgrain_grid: 512×512 d_grain distribution [m]
        """
        print("Loading φ and d_grain data...")
        
        # Read φ data
        phi_1d = np.loadtxt(phi_file)
        phi_grid = phi_1d.reshape(self.Lx, self.Ly)
        
        # Read d_grain data
        dgrain_1d = np.loadtxt(dgrain_file)
        dgrain_grid = dgrain_1d.reshape(self.Lx, self.Ly)
        
        print(f"φ value range: [{np.min(phi_grid):.4f}, {np.max(phi_grid):.4f}]")
        print(f"d_grain range: [{np.min(dgrain_grid)*1e9:.1f}, {np.max(dgrain_grid)*1e9:.1f}] nm")
        
        return phi_grid, dgrain_grid
    
    def calculate_ms_conductivity(self, phi_grid, dgrain_grid):
        """
        Calculate conductivity distribution using MS model
        
        Parameters:
        - phi_grid: 512×512 φ distribution
        - dgrain_grid: 512×512 d_grain distribution [m]
        
        Returns:
        - sigma_grid: 512×512 conductivity distribution [S/m]
        """
        print("Calculating MS conductivity distribution...")
        
        sigma_grid = np.zeros_like(phi_grid)
        alpha_grid = np.zeros_like(phi_grid)
        R_grid = np.zeros_like(phi_grid)
        
        for i in range(self.Lx):
            for j in range(self.Ly):
                phi_ij = max(phi_grid[i, j], self.material_params['phi_min'])
                d_grain_ij = dgrain_grid[i, j]
                
                # Calculate reflection coefficient
                R = self.material_params['R0'] * ((1 - phi_ij) ** 2)
                R = min(R, self.material_params['R_max'])
                R_grid[i, j] = R
                
                # Calculate scattering intensity parameter
                if R < 0.999 and d_grain_ij > 0:
                    alpha = (self.material_params['lambda_mfp'] * R / 
                            (d_grain_ij * (1 - R)))
                    alpha = min(alpha, self.material_params['alpha_max'])
                else:
                    alpha = self.material_params['alpha_max']
                
                alpha_grid[i, j] = alpha
                
                # MS conductivity formula
                sigma = self.material_params['sigma_bulk'] * (
                    1 - (3/2) * alpha / (1 + alpha)
                )
                
                # Numerical protection
                sigma_grid[i, j] = max(sigma, self.material_params['sigma_min'])
        
        print(f"Conductivity range: [{np.min(sigma_grid):.2e}, {np.max(sigma_grid):.2e}] S/m")
        print(f"Conductivity ratio range: [{np.min(sigma_grid)/self.material_params['sigma_bulk']:.6f}, {np.max(sigma_grid)/self.material_params['sigma_bulk']:.6f}]")
        
        return sigma_grid, alpha_grid, R_grid
    
    def assemble_element_matrix(self, sigma_element):
        """
        Assemble stiffness matrix for single rectangular element
        Uses 4-node bilinear element
        
        Parameters:
        - sigma_element: Element conductivity value
        
        Returns:
        - K_e: 4×4 element stiffness matrix
        """
        # Shape function derivatives for 4-node rectangular element in local coordinates
        # Node numbering:
        # 3 --- 2
        # |     |
        # 0 --- 1
        
        # Shape function derivatives (in physical coordinates)
        dN_dx = np.array([-1, 1, 1, -1]) / (2 * self.dx)  # dN/dx
        dN_dy = np.array([-1, -1, 1, 1]) / (2 * self.dy)  # dN/dy
        
        # Calculate element stiffness matrix
        K_e = np.zeros((4, 4))
        
        # Single point integration (element center)
        area = self.dx * self.dy
        
        for i in range(4):
            for j in range(4):
                # K_ij = ∫ σ (dNi/dx * dNj/dx + dNi/dy * dNj/dy) dA
                K_e[i, j] = sigma_element * (
                    dN_dx[i] * dN_dx[j] + dN_dy[i] * dN_dy[j]
                ) * area
        
        return K_e
    
    def assemble_global_matrix(self, sigma_grid):
        """
        Assemble global stiffness matrix
        
        Parameters:
        - sigma_grid: 512×512 conductivity distribution
        
        Returns:
        - K_global: Global stiffness matrix (sparse format)
        """
        print("Assembling global stiffness matrix...")
        
        # Use LIL format for assembly (more efficient)
        K_global = lil_matrix((self.n_nodes, self.n_nodes))
        
        # Iterate through all elements
        total_elements = self.Lx * self.Ly
        for i in range(self.Lx):
            for j in range(self.Ly):
                # Global node numbers for element's 4 nodes (counterclockwise)
                nodes = [
                    self.node_to_index(i, j),      # Node 0 (bottom left)
                    self.node_to_index(i, j+1),    # Node 1 (bottom right)
                    self.node_to_index(i+1, j+1),  # Node 2 (top right)
                    self.node_to_index(i+1, j)     # Node 3 (top left)
                ]
                
                # Element conductivity (use element center value)
                sigma_element = sigma_grid[i, j]
                
                # Calculate element stiffness matrix
                K_e = self.assemble_element_matrix(sigma_element)
                
                # Assemble to global matrix
                for local_i in range(4):
                    for local_j in range(4):
                        global_i = nodes[local_i]
                        global_j = nodes[local_j]
                        K_global[global_i, global_j] += K_e[local_i, local_j]
                
                # Show progress
                element_idx = i * self.Ly + j + 1
                if element_idx % 20000 == 0:
                    progress = element_idx / total_elements * 100
                    print(f"  Progress: {progress:.1f}%")
        
        print("Matrix assembly completed, keeping LIL format...")
        return K_global
    
    def apply_boundary_conditions(self, K, F, V_left=5.0, V_right=0.0):
        """
        Apply boundary conditions
        
        Parameters:
        - K: Global stiffness matrix (LIL format)
        - F: Load vector
        - V_left: Left boundary potential [V]
        - V_right: Right boundary potential [V]
        
        Returns:
        - K_bc: Stiffness matrix after applying boundary conditions (CSR format)
        - F_bc: Load vector after applying boundary conditions
        """
        print("Applying boundary conditions...")
        
        # Copy load vector
        F_bc = F.copy()
        
        # Collect boundary nodes
        boundary_nodes = []
        
        # Left boundary (j=0, displayed left): V = V_left
        for i in range(self.n_nodes_x):
            node_idx = self.node_to_index(i, 0)
            boundary_nodes.append((node_idx, V_left))
        
        # Right boundary (j=Ly, displayed right): V = V_right
        for i in range(self.n_nodes_x):
            node_idx = self.node_to_index(i, self.Ly)
            boundary_nodes.append((node_idx, V_right))
        
        print(f"Total boundary nodes: {len(boundary_nodes)}")
        print(f"Left boundary(j=0): {self.n_nodes_x} nodes set to {V_left} V")
        print(f"Right boundary(j=Ly): {self.n_nodes_x} nodes set to {V_right} V")
        print("Top/bottom boundaries(i=0,Lx): Natural boundary conditions (∂V/∂n = 0)")
        
        # Apply boundary conditions in batch (under LIL format)
        print("Modifying matrix...")
        for i, (node_idx, value) in enumerate(boundary_nodes):
            # Clear this row
            K[node_idx, :] = 0
            # Set diagonal element
            K[node_idx, node_idx] = 1
            # Set load vector
            F_bc[node_idx] = value
            
            # Show progress
            if (i + 1) % 100 == 0:
                progress = (i + 1) / len(boundary_nodes) * 100
                print(f"  Boundary condition progress: {progress:.1f}%")
        
        # Convert to CSR format for solving
        print("Converting to CSR format...")
        K_bc = K.tocsr()
        
        print("Boundary conditions applied successfully")
        return K_bc, F_bc
    
    def solve_potential(self, phi_file, dgrain_file, V_left=5.0, V_right=0.0):
        """
        Complete potential distribution solving workflow
        
        Parameters:
        - phi_file: φ data file path
        - dgrain_file: d_grain data file path
        - V_left: Left boundary potential [V]
        - V_right: Right boundary potential [V]
        
        Returns:
        - V_grid: Potential distribution (513×513)
        - results: Calculation results dictionary
        """
        start_time = time.time()
        
        # Step 1: Load data
        phi_grid, dgrain_grid = self.load_phi_and_dgrain_data(phi_file, dgrain_file)
        
        # Step 2: Calculate MS conductivity
        sigma_grid, alpha_grid, R_grid = self.calculate_ms_conductivity(phi_grid, dgrain_grid)
        
        # Step 3: Assemble global stiffness matrix
        K = self.assemble_global_matrix(sigma_grid)
        
        # Step 4: Initialize load vector
        F = np.zeros(self.n_nodes)
        
        # Step 5: Apply boundary conditions
        K_bc, F_bc = self.apply_boundary_conditions(K, F, V_left, V_right)
        
        # Step 6: Solve linear system
        print("Solving linear system...")
        solve_start = time.time()
        V_vector = spsolve(K_bc, F_bc)
        solve_time = time.time() - solve_start
        print(f"Linear solve time: {solve_time:.2f} seconds")
        
        # Step 7: Reshape to grid form (note: row-major reshape, corresponding to node_to_index mapping)
        V_grid = V_vector.reshape(self.n_nodes_x, self.n_nodes_y)
        
        # Step 8: Post-processing calculations
        results = self.post_process(V_grid, sigma_grid, phi_grid, dgrain_grid, 
                                  alpha_grid, R_grid)
        
        total_time = time.time() - start_time
        print(f"Total calculation time: {total_time:.2f} seconds")
        
        return V_grid, results
    
    def post_process(self, V_grid, sigma_grid, phi_grid, dgrain_grid, alpha_grid, R_grid):
        """
        Post-processing calculations
        
        Returns:
        - results: Dictionary containing various calculation results
        """
        print("Performing post-processing calculations...")
        
        # Electric field calculation (at element centers)
        Ex = np.zeros((self.Lx, self.Ly))
        Ey = np.zeros((self.Lx, self.Ly))
        
        for i in range(self.Lx):
            for j in range(self.Ly):
                # 4 node potentials of element (V_grid[i,j] where i is X direction, j is Y direction)
                V_nodes = [
                    V_grid[i, j],      # Bottom left
                    V_grid[i+1, j],    # Bottom right  
                    V_grid[i+1, j+1],  # Top right
                    V_grid[i, j+1]     # Top left
                ]
                
                # Calculate electric field at element center E = -∇V
                # Now j direction is horizontal (left to right), i direction is vertical
                # Ex = -∂V/∂j (horizontal electric field, left to right)
                Ex[i, j] = -(V_nodes[3] + V_nodes[2] - V_nodes[0] - V_nodes[1]) / (2 * self.dx)
                # Ey = -∂V/∂i (vertical electric field, bottom to top)  
                Ey[i, j] = -(V_nodes[1] + V_nodes[2] - V_nodes[0] - V_nodes[3]) / (2 * self.dy)
        
        # Current density calculation J = σE
        Jx = sigma_grid * Ex
        Jy = sigma_grid * Ey
        J_magnitude = np.sqrt(Jx**2 + Jy**2)
        
        # Power dissipation calculation P = J·E = σ|E|²
        E_magnitude = np.sqrt(Ex**2 + Ey**2)
        P_loss = sigma_grid * E_magnitude**2
        
        # Total current calculation (through right boundary, rightward positive)
        I_total = np.sum(Jx[:, -1]) * self.dy  # [A]
        
        # Effective conductivity calculation
        V_applied = 5.0  # Applied voltage
        L_sample = self.Lx * self.dx  # Sample length
        A_cross = self.Ly * self.dy   # Cross-sectional area
        
        sigma_eff = I_total * L_sample / (V_applied * A_cross)
        
        # Resistance enhancement factor
        rho_bulk = 1 / self.material_params['sigma_bulk']
        rho_eff = 1 / sigma_eff
        resistance_enhancement = rho_eff / rho_bulk
        
        # Statistical information
        results = {
            # Electrical distributions
            'V_grid': V_grid,
            'Ex': Ex,
            'Ey': Ey,
            'E_magnitude': E_magnitude,
            'Jx': Jx,
            'Jy': Jy,
            'J_magnitude': J_magnitude,
            'P_loss': P_loss,
            
            # MS model related
            'sigma_grid': sigma_grid,
            'alpha_grid': alpha_grid,
            'R_grid': R_grid,
            'phi_grid': phi_grid,
            'dgrain_grid': dgrain_grid,
            
            # Macroscopic properties
            'I_total': I_total,
            'sigma_eff': sigma_eff,
            'resistance_enhancement': resistance_enhancement,
            'sigma_ratio': sigma_eff / self.material_params['sigma_bulk'],
            
            # Statistics
            'V_range': [np.min(V_grid), np.max(V_grid)],
            'E_max': np.max(E_magnitude),
            'J_max': np.max(J_magnitude),
            'P_total': np.sum(P_loss) * self.dx * self.dy
        }
        
        # Print summary
        print(f"\nCalculation results summary:")
        print(f"Potential range: [{results['V_range'][0]:.3f}, {results['V_range'][1]:.3f}] V")
        print(f"Maximum electric field: {results['E_max']:.2e} V/m")
        print(f"Maximum current density: {results['J_max']:.2e} A/m²")
        print(f"Total current: {results['I_total']:.2e} A")
        print(f"Effective conductivity: {results['sigma_eff']:.2e} S/m")
        print(f"Conductivity ratio: {results['sigma_ratio']:.6f}")
        print(f"Resistance enhancement factor: {results['resistance_enhancement']:.2f}")
        print(f"Total power dissipation: {results['P_total']:.2e} W")
        
        return results
    
    def visualize_results(self, results, output_prefix):
        """
        Visualize results
        
        Parameters:
        - results: Calculation results dictionary
        - output_prefix: Output file prefix
        """
        print("Generating visualization images...")
        
        # Create large figure
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        
        # 1. φ distribution (consistent display direction with grain_growth_simulation.py)
        im1 = axes[0,0].imshow(results['phi_grid'], cmap='viridis')
        axes[0,0].set_title('φ Distribution')
        axes[0,0].set_xlabel('X Direction')
        axes[0,0].set_ylabel('Y Direction')
        plt.colorbar(im1, ax=axes[0,0])
        
        # 2. d_grain distribution
        im2 = axes[0,1].imshow((results['dgrain_grid']*1e9), cmap='plasma')
        axes[0,1].set_title('d_grain Distribution (nm)')
        axes[0,1].set_xlabel('X Direction')
        axes[0,1].set_ylabel('Y Direction')
        plt.colorbar(im2, ax=axes[0,1])
        
        # 3. Conductivity distribution (logarithmic scale)
        im3 = axes[0,2].imshow(np.log10(results['sigma_grid']), cmap='hot')
        axes[0,2].set_title('log₁₀(Conductivity) [S/m]')
        axes[0,2].set_xlabel('X Direction')
        axes[0,2].set_ylabel('Y Direction')
        plt.colorbar(im3, ax=axes[0,2])
        
        # 4. Potential distribution
        V_node_center = results['V_grid'][:-1, :-1]  # Convert to element center values for display
        im4 = axes[1,0].imshow(V_node_center, cmap='coolwarm')
        axes[1,0].set_title('Potential Distribution [V]')
        axes[1,0].set_xlabel('X Direction (Left 5V → Right 0V)')
        axes[1,0].set_ylabel('Y Direction')
        plt.colorbar(im4, ax=axes[1,0])
        
        # 5. Electric field magnitude
        im5 = axes[1,1].imshow(results['E_magnitude'], cmap='inferno')
        axes[1,1].set_title('Electric Field Magnitude [V/m]')
        axes[1,1].set_xlabel('X Direction')
        axes[1,1].set_ylabel('Y Direction')
        plt.colorbar(im5, ax=axes[1,1])
        
        # 6. Current density magnitude
        im6 = axes[1,2].imshow(results['J_magnitude'], cmap='magma')
        axes[1,2].set_title('Current Density Magnitude [A/m²]')
        axes[1,2].set_xlabel('X Direction')
        axes[1,2].set_ylabel('Y Direction')
        plt.colorbar(im6, ax=axes[1,2])
        
        # 7. Power dissipation
        im7 = axes[2,0].imshow(results['P_loss'], cmap='Reds')
        axes[2,0].set_title('Power Loss Density [W/m³]')
        axes[2,0].set_xlabel('X Direction')
        axes[2,0].set_ylabel('Y Direction')
        plt.colorbar(im7, ax=axes[2,0])
        
        # 8. Scattering parameter α
        im8 = axes[2,1].imshow(results['alpha_grid'], cmap='copper')
        axes[2,1].set_title('Scattering Parameter α')
        axes[2,1].set_xlabel('X Direction')
        axes[2,1].set_ylabel('Y Direction')
        plt.colorbar(im8, ax=axes[2,1])
        
        # 9. Reflection coefficient R
        im9 = axes[2,2].imshow(results['R_grid'], cmap='Blues')
        axes[2,2].set_title('Reflection Coefficient R')
        axes[2,2].set_xlabel('X Direction')
        axes[2,2].set_ylabel('Y Direction')
        plt.colorbar(im9, ax=axes[2,2])
        
        plt.tight_layout()
        
        # Save image
        image_file = f"{output_prefix}_fem_results.png"
        plt.savefig(image_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Results image saved: {image_file}")
    


def main():
    """Main function"""
    print("=" * 60)
    print("FEM Potential Distribution Solver Based on MS Model")
    print("=" * 60)
    
    # Set file paths
    phi_file = "grain_growth_results/time_5000.txt"
    dgrain_file = "grain_growth_results/time_5000_dgrain.txt"
    
    # Check file existence
    if not os.path.exists(phi_file):
        print(f"Error: φ file does not exist: {phi_file}")
        return
    
    if not os.path.exists(dgrain_file):
        print(f"Error: d_grain file does not exist: {dgrain_file}")
        print("Please run phi_to_dgrain_converter.py first to generate d_grain file")
        return
    
    # Create FEM solver
    solver = FEMPotentialSolver(Lx=512, Ly=512, dx=2.0e-6)
    
    # Solve potential distribution
    try:
        V_grid, results = solver.solve_potential(
            phi_file, dgrain_file, V_left=5.0, V_right=0.0
        )
        
        # Visualize results
        output_prefix = "grain_growth_results/time_5000"
        solver.visualize_results(results, output_prefix)
        
        print("\n" + "=" * 60)
        print("FEM solving completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error occurred during solving: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
