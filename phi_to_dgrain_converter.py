import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import label
import os

def load_phi_data(phi_file):
    """
    Load φ data file (1 column, 512*512 rows)
    Returns 512×512 2D array
    """
    print(f"Reading file: {phi_file}")
    
    # Read single column data
    phi_1d = np.loadtxt(phi_file)
    print(f"Read {len(phi_1d)} data points")
    
    # Reshape to 512×512 matrix
    if len(phi_1d) != 512 * 512:
        raise ValueError(f"Number of data points {len(phi_1d)} not equal to 512×512 = {512*512}")
    
    # Reshape to 512×512 matrix, maintaining consistent order with grain_growth_simulation.py
    # grain_growth_simulation.py uses: for IX in range(Lx): for IY in range(Ly): FAI[IX, IY]
    phi_2d = phi_1d.reshape(512, 512)
    
    # Display basic statistics
    print(f"φ value range: [{np.min(phi_2d):.6f}, {np.max(phi_2d):.6f}]")
    print(f"φ value mean: {np.mean(phi_2d):.6f}")
    print(f"φ value std dev: {np.std(phi_2d):.6f}")
    
    return phi_2d

def analyze_phi_distribution(phi_2d):
    """Analyze φ distribution and determine appropriate threshold"""
    
    # Calculate statistics
    phi_min = np.min(phi_2d)
    phi_max = np.max(phi_2d)
    phi_mean = np.mean(phi_2d)
    phi_std = np.std(phi_2d)
    
    # Calculate percentiles
    percentiles = np.percentile(phi_2d, [10, 25, 50, 75, 90, 95, 99])
    
    print("\nφ distribution statistical analysis:")
    print(f"Minimum: {phi_min:.6f}")
    print(f"Maximum: {phi_max:.6f}")
    print(f"Mean: {phi_mean:.6f}")
    print(f"Standard deviation: {phi_std:.6f}")
    print(f"Percentiles [10%, 25%, 50%, 75%, 90%, 95%, 99%]: {percentiles}")
    
    # Determine grain identification threshold
    # Use 75% percentile or 0.8, whichever is smaller
    grain_threshold = min(0.8, percentiles[3])  # 75th percentile
    grain_threshold = max(grain_threshold, phi_mean + 0.5 * phi_std)
    
    print(f"\nRecommended grain identification threshold: {grain_threshold:.4f}")
    
    return grain_threshold

def calculate_grain_properties(phi_2d, grain_threshold, dx):
    """
    Calculate grain properties
    """
    print(f"\nUsing threshold {grain_threshold:.4f} for grain identification...")
    
    # Grain identification
    grain_mask = phi_2d > grain_threshold
    labeled_grains, num_grains = label(grain_mask)
    
    print(f"Identified {num_grains} grains")
    
    if num_grains == 0:
        print("Warning: No grains identified, retrying with lower threshold...")
        grain_threshold = np.percentile(phi_2d, 90)  # Use 90% percentile
        print(f"Using new threshold: {grain_threshold:.4f}")
        grain_mask = phi_2d > grain_threshold
        labeled_grains, num_grains = label(grain_mask)
        print(f"Re-identified {num_grains} grains")
    
    # Calculate properties for each grain
    grain_properties = {}
    all_diameters = []
    
    for grain_id in range(1, num_grains + 1):
        # Get grain pixels
        grain_pixels = (labeled_grains == grain_id)
        area_pixels = np.sum(grain_pixels)
        
        # Skip grains that are too small (likely noise)
        if area_pixels < 10:  # Less than 10 pixels
            continue
            
        # Calculate physical area and equivalent diameter
        area_physical = area_pixels * (dx ** 2)  # Physical area [m²]
        d_equivalent = 2 * np.sqrt(area_physical / np.pi)  # Equivalent diameter [m]
        
        # Calculate average φ value within grain
        phi_avg = np.mean(phi_2d[grain_pixels])
        
        # Calculate grain bounding box
        coords = np.where(grain_pixels)
        bbox_height = (np.max(coords[0]) - np.min(coords[0]) + 1) * dx
        bbox_width = (np.max(coords[1]) - np.min(coords[1]) + 1) * dx
        
        grain_properties[grain_id] = {
            'diameter': d_equivalent,
            'area': area_physical,
            'phi_avg': phi_avg,
            'pixel_count': area_pixels,
            'bbox_height': bbox_height,
            'bbox_width': bbox_width
        }
        
        all_diameters.append(d_equivalent)
    
    # Calculate global statistics
    if all_diameters:
        d0_global = np.mean(all_diameters)
        d_std = np.std(all_diameters)
        d_min = np.min(all_diameters)
        d_max = np.max(all_diameters)
        
        print(f"\nGrain size statistics:")
        print(f"Number of valid grains: {len(all_diameters)}")
        print(f"Average diameter: {d0_global*1e9:.1f} nm")
        print(f"Standard deviation: {d_std*1e9:.1f} nm")
        print(f"Diameter range: [{d_min*1e9:.1f}, {d_max*1e9:.1f}] nm")
    else:
        print("Warning: No valid grains found!")
        d0_global = 50e-9  # Default value 50nm
    
    return labeled_grains, grain_properties, d0_global

def generate_dgrain_distribution(phi_2d, labeled_grains, grain_properties, d0_global):
    """
    Generate 512×512 d_grain distribution
    """
    print("\nGenerating d_grain distribution...")
    
    d_grain_grid = np.zeros_like(phi_2d, dtype=np.float64)
    
    rows, cols = phi_2d.shape
    
    for i in range(rows):
        for j in range(cols):
            phi_ij = phi_2d[i, j]
            grain_id = labeled_grains[i, j]
            
            if grain_id > 0 and grain_id in grain_properties:
                # Inside identified grain
                d_local = grain_properties[grain_id]['diameter']
                phi_avg = grain_properties[grain_id]['phi_avg']
                
                # φ correction factor
                if phi_avg > 0:
                    correction = (phi_ij / phi_avg) ** (1/3)
                    # Limit correction range to avoid extreme values
                    correction = np.clip(correction, 0.3, 3.0)
                else:
                    correction = 1.0
                
                d_grain_grid[i, j] = d_local * correction
                
            else:
                # In grain boundary or unidentified region
                if phi_ij > 0.1:  # Some degree of order
                    d_grain_grid[i, j] = d0_global * (phi_ij ** (1/3))
                else:  # Grain boundary or disordered region
                    d_grain_grid[i, j] = d0_global * 0.1  # Very small effective size
    
    # Apply light Gaussian smoothing to reduce discontinuities
    print("Applying smoothing...")
    d_grain_smooth = ndimage.gaussian_filter(d_grain_grid, sigma=0.8)
    
    # Apply smoothing only in grain boundary regions, keep grain centers unchanged
    grain_core_mask = phi_2d > (np.percentile(phi_2d, 75) + 0.05)
    d_grain_final = np.where(grain_core_mask, d_grain_grid, d_grain_smooth)
    
    # Ensure no zero or negative values
    d_min_safe = d0_global * 0.01  # Minimum safe value
    d_grain_final = np.maximum(d_grain_final, d_min_safe)
    
    print(f"d_grain final range: [{np.min(d_grain_final)*1e9:.1f}, {np.max(d_grain_final)*1e9:.1f}] nm")
    print(f"d_grain average: {np.mean(d_grain_final)*1e9:.1f} nm")
    
    return d_grain_final

def save_dgrain_file(d_grain_2d, output_file):
    """
    Save d_grain data as single column format (512*512 rows)
    Same format as original φ file
    """
    print(f"\nSaving d_grain data to: {output_file}")
    
    # Flatten 2D array to 1D
    d_grain_1d = d_grain_2d.flatten()
    
    # Validate data size
    if len(d_grain_1d) != 512 * 512:
        raise ValueError(f"Incorrect number of data points: {len(d_grain_1d)} != {512*512}")
    
    # Save as single column format, maintaining same precision as original file
    with open(output_file, 'w') as f:
        for value in d_grain_1d:
            f.write(f'{value:14.6e}\n')  # Scientific notation, 6 decimal places
    
    print(f"Successfully saved {len(d_grain_1d)} d_grain values")

def create_visualization(phi_2d, d_grain_2d, labeled_grains, output_prefix):
    """
    Create visualization images
    """
    print("\nCreating visualization images...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # φ distribution (no transpose, consistent with grain_growth_simulation.py)
    im1 = axes[0,0].imshow(phi_2d, cmap='viridis')
    axes[0,0].set_title('φ Distribution')
    axes[0,0].set_xlabel('X')
    axes[0,0].set_ylabel('Y')
    plt.colorbar(im1, ax=axes[0,0])
    
    # d_grain distribution (in nm, no transpose)
    im2 = axes[0,1].imshow(d_grain_2d * 1e9, cmap='plasma')
    axes[0,1].set_title('d_grain Distribution (nm)')
    axes[0,1].set_xlabel('X')
    axes[0,1].set_ylabel('Y')
    plt.colorbar(im2, ax=axes[0,1])
    
    # Labeled grains (no transpose)
    im3 = axes[1,0].imshow(labeled_grains, cmap='tab20')
    axes[1,0].set_title(f'Labeled Grains ({np.max(labeled_grains)} grains)')
    axes[1,0].set_xlabel('X')
    axes[1,0].set_ylabel('Y')
    plt.colorbar(im3, ax=axes[1,0])
    
    # d_grain histogram
    axes[1,1].hist(d_grain_2d.flatten() * 1e9, bins=50, alpha=0.7, edgecolor='black')
    axes[1,1].set_xlabel('d_grain (nm)')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('d_grain Distribution Histogram')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # save image
    image_file = f"{output_prefix}_analysis.png"
    plt.savefig(image_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization image saved: {image_file}")

def main(phi_file, dx=2.0e-6, output_file=None):
    """
    Main function: Convert φ file to d_grain file
    
    Parameters:
    - phi_file: Input φ data file path
    - dx: Grid spacing [m] (default 2 microns)
    - output_file: Output file path (if None, auto-generated)
    """
    
    print("=" * 60)
    print("φ value to d_grain value conversion program")
    print("=" * 60)
    
    # Check input file
    if not os.path.exists(phi_file):
        raise FileNotFoundError(f"Input file does not exist: {phi_file}")
    
    # Generate output filename
    if output_file is None:
        base_name = os.path.splitext(phi_file)[0]
        output_file = f"{base_name}_dgrain.txt"
    
    # Step 1: Read φ data
    phi_2d = load_phi_data(phi_file)
    
    # Step 2: Analyze φ distribution
    grain_threshold = analyze_phi_distribution(phi_2d)
    
    # Step 3: Grain analysis
    labeled_grains, grain_properties, d0_global = calculate_grain_properties(
        phi_2d, grain_threshold, dx)
    
    # Step 4: Generate d_grain distribution
    d_grain_2d = generate_dgrain_distribution(
        phi_2d, labeled_grains, grain_properties, d0_global)
    
    # Step 5: Save d_grain file
    save_dgrain_file(d_grain_2d, output_file)
    
    # Step 6: Create visualization
    output_prefix = os.path.splitext(output_file)[0]
    create_visualization(phi_2d, d_grain_2d, labeled_grains, output_prefix)
    
    # Output summary
    print("\n" + "=" * 60)
    print("Conversion completed summary:")
    print(f"Input file: {phi_file}")
    print(f"Output file: {output_file}")
    print(f"Grid size: 512 × 512")
    print(f"Grid spacing: {dx*1e6:.1f} μm")
    print(f"Identified grains: {len(grain_properties)}")
    print(f"Average grain size: {d0_global*1e9:.1f} nm")
    print(f"d_grain range: [{np.min(d_grain_2d)*1e9:.1f}, {np.max(d_grain_2d)*1e9:.1f}] nm")
    print("=" * 60)
    
    return d_grain_2d, grain_properties

if __name__ == "__main__":
    # Usage example
    phi_file = "grain_growth_results/time_5000.txt"
    
    # Material parameters
    dx = 2.0e-6  # Grid spacing, assumed to be 2 microns
    
    # Run conversion
    try:
        d_grain_2d, grain_info = main(phi_file, dx)
        print("\nProgram executed successfully!")
    except Exception as e:
        print(f"\nError: {e}")
        print("Please check input file path and format")
