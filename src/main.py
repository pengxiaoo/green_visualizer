import os
from pathlib import Path
from green_visualizer import GreenVisualizer

def process_json_file(json_path, output_path):
    """Process a single JSON file and save the visualization"""
    try:
        visualizer = GreenVisualizer(json_path)
        visualizer.save_plot(output_path)
        print(f"Successfully processed: {json_path} -> {output_path}")
    except Exception as e:
        print(f"Error processing {json_path}: {str(e)}")

def main():
    # Define paths
    base_dir = Path(__file__).parent.parent  # Get project root directory
    json_dir = base_dir / "testcases" / "json"
    output_dir = base_dir / "testcases" / "map"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process all JSON files
    for json_file in json_dir.glob("*.json"):
        # Create output path with same name but .png extension
        output_path = output_dir / f"{json_file.stem}.png"
        process_json_file(str(json_file), str(output_path))

if __name__ == "__main__":
    main() 