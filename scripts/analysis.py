#!/usr/bin/env python3

import os
import json
import argparse
import numpy as np
from pathlib import Path

def find_json_files(directory):
    """Recursively find all stop_data.json files in the given directory."""
    json_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == 'stop_data.json':
                json_files.append(os.path.join(root, file))
    return json_files

def extract_stopping_time(json_file):
    """Extract stopping_time_ms from a JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data.get('stopping_time_ms')

def analyze_stopping_times(stopping_times):
    """Calculate basic statistics for stopping times."""
    stopping_times = np.array(stopping_times)
    stats = {
        'count': len(stopping_times),
        'mean': np.mean(stopping_times),
        'std': np.std(stopping_times),
        'min': np.min(stopping_times),
        'max': np.max(stopping_times),
        'median': np.median(stopping_times)
    }
    return stats

def format_for_latex(stats, test_condition):
    """Format the statistics for a LaTeX table."""
    latex_table = (
        f"\\begin{{table}}[h]\n"
        f"\\centering\n"
        f"\\caption{{Stopping Time Statistics for {test_condition}}}\n"
        f"\\begin{{tabular}}{{|l|r|}} \\hline\n"
        f"Metric & Value \\\\ \\hline\n"
        f"Number of samples & {stats['count']} \\\\\n"
        f"Mean stopping time (ms) & {stats['mean']:.4f} \\\\\n"
        f"Std Dev stopping time (ms) & {stats['std']:.4f} \\\\\n"
        f"Min stopping time (ms) & {stats['min']:.4f} \\\\\n"
        f"Max stopping time (ms) & {stats['max']:.4f} \\\\\n"
        f"Median stopping time (ms) & {stats['median']:.4f} \\\\\n"
        f"\\hline\n"
        f"\\end{{tabular}}\n"
        f"\\label{{tab:stopping_time_stats}}\n"
        f"\\end{{table}}"
    )
    return latex_table

def main():
    parser = argparse.ArgumentParser(description='Analyze HAAM experiment stopping times.')
    parser.add_argument('directory', type=str, help='Directory containing experiment logs (e.g., "2000mm")')
    args = parser.parse_args()
    
    # Get the test condition name from the directory path
    test_condition = os.path.basename(os.path.normpath(args.directory))
    print(f"Analyzing stopping times for test condition: {test_condition}")
    
    # Find all JSON files
    json_files = find_json_files(args.directory)
    print(f"Found {len(json_files)} JSON files.")
    
    # Extract stopping times from JSON files
    stopping_times = []
    for json_file in json_files:
        try:
            stopping_time = extract_stopping_time(json_file)
            if stopping_time is not None:
                stopping_times.append(stopping_time)
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    if not stopping_times:
        print("No valid stopping time data found. Exiting.")
        return
        
    # Analyze the stopping times
    stats = analyze_stopping_times(stopping_times)
    
    # Print summary statistics
    print(f"\nStopping Time Statistics for {test_condition}:")
    for key, value in stats.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # Format for LaTeX
    latex_table = format_for_latex(stats, test_condition)
    
    # Create output directory
    output_dir = Path(args.directory) / "analysis_results"
    output_dir.mkdir(exist_ok=True)
    
    # Save the raw stopping times to a text file
    with open(output_dir / "stopping_times.txt", "w") as f:
        for time in stopping_times:
            f.write(f"{time}\n")
    
    # Save the LaTeX formatted table
    with open(output_dir / "latex_table.txt", "w") as f:
        f.write(latex_table)
    
    print(f"\nResults saved to {output_dir}")
    print("\nLaTeX Table:")
    print(latex_table)
    
    # Print raw data
    print("\nRaw stopping times (ms):")
    for time in stopping_times[:10]:  # Print first 10 for brevity
        print(f"{time:.4f}")
    if len(stopping_times) > 10:
        print("...")

if __name__ == "__main__":
    main()