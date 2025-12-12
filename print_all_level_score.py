import argparse
import json
import sys
import os
import numpy as np

def process_scores(jsonl_file, output_file):
    """
    Reads all_level_scores from a .jsonl file, calculates mean and std per level,
    and saves the results to a JSON file.
    """
    if not os.path.exists(jsonl_file):
        print(f"Error: File not found: {jsonl_file}", file=sys.stderr)
        return

    print(f"--- Reading file: {jsonl_file} ---")
    
    # Dictionary to store all scores for each level. Key: level index, Value: list of all scores
    level_scores_map = {} 

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
                        if level_idx not in level_scores_map:
                            level_scores_map[level_idx] = []
                        # scores is a list of floats for that level
                        level_scores_map[level_idx].extend(scores)
                        
                except json.JSONDecodeError:
                    print(f"Error: Invalid JSON at line {line_number}", file=sys.stderr)
                except Exception as e:
                    print(f"Error processing line {line_number}: {e}", file=sys.stderr)

        # Calculate mean and std
        sorted_levels = sorted(level_scores_map.keys())
        
        means = []
        stds = []
        levels = []

        print("\n--- Statistics per Level ---")
        for level in sorted_levels:
            scores = level_scores_map[level]
            if scores:
                mean_val = np.mean(scores)
                std_val = np.std(scores)
                means.append(float(mean_val))
                stds.append(float(std_val))
                levels.append(level)
                print(f"Level {level}: Mean = {mean_val:.4f}, Std = {std_val:.4f}, Count = {len(scores)}")
            else:
                means.append(0.0)
                stds.append(0.0)
                levels.append(level)

        output_data = {
            "levels": levels,
            "means": means,
            "stds": stds
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
