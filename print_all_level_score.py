import argparse
import json
import sys
import os
import numpy as np

def process_scores(jsonl_file, output_file):
    """
    Reads all_level_scores from a .jsonl file.
    Calculates mean and std for EACH sample first, then averages them.
    This isolates intra-sample variance from inter-sample variance.
    """
    if not os.path.exists(jsonl_file):
        print(f"Error: File not found: {jsonl_file}", file=sys.stderr)
        return

    print(f"--- Reading file: {jsonl_file} ---")
    
    # Dictionary to store lists of means and stds for each level
    # Key: level index, Value: {'means': [], 'stds': []}
    level_stats_map = {} 

    try:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    # all_level_scores is expected to be a list of lists of floats
                    # e.g. [[score1, score2], [score3, score4], ...]
                    all_level_scores = data.get('all_level_scores', None)
                    
                    if all_level_scores is None:
                        continue
                    
                    for level_idx, scores in enumerate(all_level_scores):
                        if not scores:
                            continue

                        if level_idx not in level_stats_map:
                            level_stats_map[level_idx] = {'means': [], 'stds': []}
                        
                        # Calculate stats for this specific sample at this level
                        # This captures the spread of beams for THIS problem only
                        sample_mean = np.mean(scores)
                        sample_std = np.std(scores)
                        
                        level_stats_map[level_idx]['means'].append(sample_mean)
                        level_stats_map[level_idx]['stds'].append(sample_std)
                        
                except json.JSONDecodeError:
                    print(f"Error: Invalid JSON at line {line_number}", file=sys.stderr)
                except Exception as e:
                    print(f"Error processing line {line_number}: {e}", file=sys.stderr)

        # Calculate average of means and average of stds
        sorted_levels = sorted(level_stats_map.keys())
        
        final_means = []
        final_stds = []
        levels = []

        print("\n--- Statistics per Level (Averaged Intra-Sample Stats) ---")
        for level in sorted_levels:
            sample_means = level_stats_map[level]['means']
            sample_stds = level_stats_map[level]['stds']
            
            if sample_means:
                # The average of per-sample means
                avg_mean_val = np.mean(sample_means)
                # The average of per-sample stds (Average Intra-Sample Std)
                # This tells us: "On average, how much do beams diverge within a single problem?"
                avg_std_val = np.mean(sample_stds)
                
                final_means.append(float(avg_mean_val))
                final_stds.append(float(avg_std_val))
                levels.append(level)
                print(f"Level {level}: Mean = {avg_mean_val:.4f}, Avg Intra-Sample Std = {avg_std_val:.4f}, Count = {len(sample_means)}")
            else:
                final_means.append(0.0)
                final_stds.append(0.0)
                levels.append(level)

        output_data = {
            "levels": levels,
            "means": final_means,
            "stds": final_stds
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4)
        
        print(f"\nStatistics saved to {output_file}")

    except IOError as e:
        print(f"Error reading file: {e}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description="Read all_level_scores from jsonl and calculate stats.")
    parser.add_argument("jsonl_file", type=str, help="Path to input .jsonl file")
    parser.add_argument("--output", type=str, default="level_stats.json", help="Path to output JSON file for plotting")
    
    args = parser.parse_args()
    process_scores(args.jsonl_file, args.output)

if __name__ == "__main__":
    main()
