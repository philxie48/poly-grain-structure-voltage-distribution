import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend, no window display
import matplotlib.pyplot as plt
import os
import shutil
from tqdm import tqdm
import time

def grain_growth_simulation():
    """
    Grain growth simulation using the phase field method.
    Converted from MATLAB code to Python.
    """
    # Create output directory
    output_dir = os.path.join(os.getcwd(), "grain_growth_results")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    # Simulation parameters
    P = 30      # Number of grain orientations
    Lx = 512    # Number of grid points in x direction
    Ly = 512    # Number of grid points in y direction
    DeltaX = 2.0  # Grid size
    DeltaT = 0.25  # Time step
    
    nprint = 1  # Print interval
    
    # Model parameters
    Alpha = 1.0
    Beta = 1.0
    Gamar = 1.0
    Ki = 2.0
    L = 1.0
    
    # Initialize arrays
    ITA = np.random.rand(Lx, Ly, P) * 0.2 - 0.1  # Initial random values for orientations
    ITANEW = np.zeros((Lx, Ly, P))
    FAI = np.zeros((Lx, Ly))
    TADD = np.zeros((Lx, Ly, P))
    DELTAPF = np.zeros((Lx, Ly, P))
    
    # Total simulation time steps
    T_TIME = 5000
    
    # Main simulation loop
    for istep in tqdm(range(1, T_TIME + 1), desc="Grain Growth Simulation"):
        # Apply boundary conditions and compute phase field
        for II in range(Lx):
            for JJ in range(Ly):
                # Periodic boundary conditions
                JJP = (JJ + 1) % Ly
                JJM = (JJ - 1) % Ly
                IIP = (II + 1) % Lx
                IIM = (II - 1) % Lx
                
                # Compute phase field equations
                for KK in range(P):
                    TADD[II, JJ, KK] = 0.0
                    for KKK in range(P):
                        if KKK != KK:
                            TADD[II, JJ, KK] += ITA[II, JJ, KKK]**2
                    
                    # Compute Laplacian (using 9-point stencil)
                    DELTAPF[II, JJ, KK] = (1/DeltaX**2) * (
                        0.5 * (ITA[IIM, JJ, KK] + ITA[IIP, JJ, KK] + 
                              ITA[II, JJM, KK] + ITA[II, JJP, KK] - 
                              4 * ITA[II, JJ, KK]) + 
                        0.25 * (ITA[IIM, JJM, KK] + ITA[IIM, JJP, KK] + 
                               ITA[IIP, JJM, KK] + ITA[IIP, JJP, KK] - 
                               4 * ITA[II, JJ, KK])
                    )
        
        # Update phase field (vectorized operation)
        ITANEW = ITA + DeltaT * (-L) * (
            -Alpha * ITA + Beta * ITA**3 + 2 * Gamar * ITA * TADD - Ki * DELTAPF
        )
        
        # Update ITA for next iteration
        ITA = ITANEW.copy()
        
        # Display and save results at specified intervals
        if (istep % nprint == 0) or (istep == 1):
            print(f'done step: {istep:5d}')
            
            # Calculate order parameter (FAI)
            for IX in range(Lx):
                for IY in range(Ly):
                    TFAI = 0.0
                    for IK in range(P):
                        TFAI += ITA[IX, IY, IK]**2
                    FAI[IX, IY] = TFAI
            
            # No display - only saving data and images
        
        # Save data and figures at larger intervals
        if (istep % (nprint * 100) == 0) or (istep == 1):
            # Save data to text file
            fname = os.path.join(output_dir, f'time_{istep}.txt')
            with open(fname, 'w') as f:
                for IX in range(Lx):
                    for IY in range(Ly):
                        f.write(f'{FAI[IX, IY]:14.6f}\n')
            
            # Save figure as JPG
            plt.figure(figsize=(10, 10))
            plt.imshow(FAI, cmap='viridis')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{istep}.jpg'), dpi=150, bbox_inches='tight')
            plt.close()
    
    print(f"Simulation completed. Results saved to {output_dir}")

def faster_grain_growth_simulation():
    """
    Optimized version of the grain growth simulation using vectorized operations.
    """
    # Create output directory
    output_dir = os.path.join(os.getcwd(), "grain_growth_results")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    # Simulation parameters
    P = 30      # Number of grain orientations
    Lx = 512    # Number of grid points in x direction
    Ly = 512    # Number of grid points in y direction
    DeltaX = 2.0  # Grid size
    DeltaT = 0.25  # Time step
    
    nprint = 1  # Print interval
    
    # Model parameters
    Alpha = 1.0
    Beta = 1.0
    Gamar = 1.0
    Ki = 2.0
    L = 1.0
    
    # Initialize arrays
    ITA = np.random.rand(Lx, Ly, P) * 0.2 - 0.1  # Initial random values for orientations
    
    # Total simulation time steps
    T_TIME = 5000
    
    # Main simulation loop
    for istep in tqdm(range(1, T_TIME + 1), desc="Grain Growth Simulation"):
        # Calculate TADD (sum of squared ITA for other orientations)
        TADD = np.zeros_like(ITA)
        for KK in range(P):
            # Sum of all orientations squared
            total_sum = np.sum(ITA**2, axis=2)
            # Subtract current orientation squared
            TADD[:, :, KK] = total_sum - ITA[:, :, KK]**2
        
        # Calculate Laplacian using periodic boundary conditions
        DELTAPF = np.zeros_like(ITA)
        for KK in range(P):
            # Shifted arrays for neighbors
            ITA_left = np.roll(ITA[:, :, KK], 1, axis=0)
            ITA_right = np.roll(ITA[:, :, KK], -1, axis=0)
            ITA_up = np.roll(ITA[:, :, KK], 1, axis=1)
            ITA_down = np.roll(ITA[:, :, KK], -1, axis=1)
            
            # Diagonal neighbors
            ITA_upleft = np.roll(np.roll(ITA[:, :, KK], 1, axis=0), 1, axis=1)
            ITA_upright = np.roll(np.roll(ITA[:, :, KK], -1, axis=0), 1, axis=1)
            ITA_downleft = np.roll(np.roll(ITA[:, :, KK], 1, axis=0), -1, axis=1)
            ITA_downright = np.roll(np.roll(ITA[:, :, KK], -1, axis=0), -1, axis=1)
            
            # 9-point stencil Laplacian
            DELTAPF[:, :, KK] = (1/DeltaX**2) * (
                0.5 * (ITA_left + ITA_right + ITA_up + ITA_down - 4 * ITA[:, :, KK]) +
                0.25 * (ITA_upleft + ITA_upright + ITA_downleft + ITA_downright - 4 * ITA[:, :, KK])
            )
        
        # Update phase field (vectorized operation)
        ITANEW = ITA + DeltaT * (-L) * (
            -Alpha * ITA + Beta * ITA**3 + 2 * Gamar * ITA * TADD - Ki * DELTAPF
        )
        
        # Update ITA for next iteration
        ITA = ITANEW.copy()
        
        # Display and save results at specified intervals
        if (istep % nprint == 0) or (istep == 1):
            print(f'done step: {istep:5d}')
            
            # Calculate order parameter (FAI) - sum of squared ITA across all orientations
            FAI = np.sum(ITA**2, axis=2)
            
            # No display - only saving data and images
        
        # Save data and figures at larger intervals
        if (istep % (nprint * 100) == 0) or (istep == 1):
            # Calculate FAI for saving
            FAI = np.sum(ITA**2, axis=2)
            
            # Save data to text file
            fname = os.path.join(output_dir, f'time_{istep}.txt')
            with open(fname, 'w') as f:
                for IX in range(Lx):
                    for IY in range(Ly):
                        f.write(f'{FAI[IX, IY]:14.6f}\n')
            
            # Save figure as JPG
            plt.figure(figsize=(10, 10))
            plt.imshow(FAI, cmap='viridis')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{istep}.jpg'), dpi=150, bbox_inches='tight')
            plt.close()
    
    print(f"Simulation completed. Results saved to {output_dir}")

if __name__ == "__main__":
    print("Starting grain growth simulation...")
    print("Using faster vectorized implementation...")
    start_time = time.time()
    faster_grain_growth_simulation()
    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds")
