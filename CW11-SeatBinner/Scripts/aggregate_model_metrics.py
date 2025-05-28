# filepath: c:\Users\arrow\Documents\GitHub\WMA\CW11-SeatBinner\aggregate_model_metrics.py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from pathlib import Path

def create_model_comparison_grid(models_dir, figure_name, output_dir="Figures"):
    """
    Create a 3x2 grid comparing model metrics across different YOLOv5 variants.
    
    Args:
        models_dir: Path to the Models directory
        figure_name: Name of the figure to compare (e.g., 'results.png', 'confusion_matrix.png')
        output_dir: Directory to save the comparison plots
    """
    
    # Model configurations: (size, training_type)
    model_configs = [
        ('n', 'pretrained'), ('s', 'pretrained'), ('m', 'pretrained'),
        ('n', 'scratch'), ('s', 'scratch'), ('m', 'scratch')
    ]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with 3x2 subplot grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Model Comparison: {figure_name.replace(".png", "").replace("_", " ").title()}', 
                 fontsize=16, fontweight='bold')
    
    for idx, (size, training_type) in enumerate(model_configs):
        row = idx // 3
        col = idx % 3
        
        # Construct path to the figure
        model_dir = f"yolov5{size}_{training_type}"
        figure_path = Path(models_dir) / model_dir / figure_name
        
        ax = axes[row, col]
        
        if figure_path.exists():
            try:
                # Load and display the image
                img = mpimg.imread(figure_path)
                ax.imshow(img)
                ax.axis('off')
                
                # Add title with model info
                title = f"YOLOv5{size.upper()} ({training_type.title()})"
                ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Error loading\n{model_dir}\n{figure_name}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"YOLOv5{size.upper()} ({training_type.title()})", 
                           fontsize=12, fontweight='bold', pad=10)
        else:
            # Display placeholder if file doesn't exist
            ax.text(0.5, 0.5, f'File not found:\n{model_dir}\n{figure_name}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"YOLOv5{size.upper()} ({training_type.title()})", 
                       fontsize=12, fontweight='bold', pad=10)
    
    # Adjust layout and save
    plt.tight_layout()
    output_path = Path(output_dir) / f"comparison_{figure_name}"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Saved comparison plot: {output_path}")

def generate_all_comparisons(models_dir="Models", output_dir="Figures"):
    """Generate comparison plots for all key metrics."""
    
    # Key figures to compare
    key_figures = [
        'results.png',           # Training metrics over time
        'confusion_matrix.png',  # Confusion matrix
        'confusion_matrix_normalized.png',  # Normalized confusion matrix
        'F1_curve.png',         # F1 score curve
        'PR_curve.png',         # Precision-Recall curve
        'P_curve.png',          # Precision curve
        'R_curve.png'           # Recall curve
    ]
    
    print("Generating model comparison plots...")
    
    for figure in key_figures:
        print(f"\nProcessing: {figure}")
        create_model_comparison_grid(models_dir, figure, output_dir)
    
    print("\nAll comparison plots generated successfully!")

if __name__ == "__main__":
    # Run from the Models directory or specify the path
    models_directory = "Models"  # Adjust path if needed
    output_directory = "Figures"  # Save to Figures directory at project root
    
    # Check if Models directory exists
    if not os.path.exists(models_directory):
        print(f"Models directory not found at: {models_directory}")
        print("Please run this script from the correct directory or update the path.")
        exit(1)
    
    generate_all_comparisons(models_directory, output_directory)