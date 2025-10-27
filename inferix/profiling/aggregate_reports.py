#!/usr/bin/env python3
"""Module to aggregate profiling reports from multiple ranks in distributed training."""

import os
import json
import argparse
from typing import Dict, Any, List
from pathlib import Path


def load_json_report(report_path: str) -> Dict[str, Any]:
    """Load a JSON profiling report.
    
    Args:
        report_path: Path to the JSON report file
        
    Returns:
        Dictionary containing the report data
    """
    with open(report_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def aggregate_metrics(metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate metrics from multiple ranks.
    
    Args:
        metrics_list: List of metric dictionaries from different ranks
        
    Returns:
        Dictionary with aggregated metrics (min, max, avg, sum)
    """
    if not metrics_list:
        return {}
    
    # For numerical values, compute min, max, avg
    aggregated = {}
    keys = metrics_list[0].keys()
    
    for key in keys:
        values = [m[key] for m in metrics_list if key in m and isinstance(m[key], (int, float))]
        if values:
            aggregated[f"{key}_min"] = min(values)
            aggregated[f"{key}_max"] = max(values)
            aggregated[f"{key}_avg"] = sum(values) / len(values)
            aggregated[f"{key}_sum"] = sum(values)
    
    return aggregated


def aggregate_reports(reports_dir: str, output_path: str):
    """Aggregate profiling reports from all ranks.
    
    Args:
        reports_dir: Directory containing rank subdirectories with profiling reports
        output_path: Output path for aggregated report
    """
    reports_path = Path(reports_dir)
    
    if not reports_path.exists():
        print(f"Reports directory {reports_dir} does not exist")
        return
    
    # Find all rank directories
    rank_dirs = [d for d in reports_path.iterdir() if d.is_dir() and d.name.startswith('rank_')]
    
    if not rank_dirs:
        print("No rank directories found")
        return
    
    print(f"Found {len(rank_dirs)} rank directories")
    
    # Collect all JSON reports
    all_reports = []
    gpu_metrics = []
    cpu_metrics = []
    
    for rank_dir in rank_dirs:
        json_reports = list(rank_dir.glob("*.json"))
        for report_path in json_reports:
            try:
                report = load_json_report(str(report_path))  # Convert Path to str
                all_reports.append(report)
                
                # Extract metrics
                if 'gpu' in report:
                    gpu_metrics.append(report['gpu'])
                if 'cpu' in report:
                    cpu_metrics.append(report['cpu'])
                    
            except Exception as e:
                print(f"Error loading report {report_path}: {e}")
    
    if not all_reports:
        print("No valid reports found")
        return
    
    # Aggregate metrics
    aggregated_data = {
        "total_reports": len(all_reports),
        "rank_count": len(rank_dirs),
        "aggregated_gpu_metrics": aggregate_metrics(gpu_metrics),
        "aggregated_cpu_metrics": aggregate_metrics(cpu_metrics),
        "reports_summary": []
    }
    
    # Add summary for each report
    for i, report in enumerate(all_reports):
        summary = {
            "report_index": i,
            "session_id": report.get("session_id", "unknown"),
            "duration": report.get("duration", 0),
            "rank": report.get("tags", {}).get("rank", "unknown")
        }
        aggregated_data["reports_summary"].append(summary)
    
    # Save aggregated report
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(aggregated_data, f, indent=2, ensure_ascii=False)
    
    print(f"Aggregated report saved to {output_path}")
    
    # Print summary
    print("\n=== Aggregation Summary ===")
    print(f"Total reports processed: {len(all_reports)}")
    print(f"Ranks: {len(rank_dirs)}")
    
    if gpu_metrics:
        print("\n=== GPU Metrics (Aggregated) ===")
        gpu_agg = aggregated_data["aggregated_gpu_metrics"]
        print(f"GPU Memory - Min: {gpu_agg.get('memory_used_mb_min', 0):.1f}MB, "
              f"Max: {gpu_agg.get('memory_used_mb_max', 0):.1f}MB, "
              f"Avg: {gpu_agg.get('memory_used_mb_avg', 0):.1f}MB")
        print(f"GPU Utilization - Min: {gpu_agg.get('utilization_percent_min', 0):.1f}%, "
              f"Max: {gpu_agg.get('utilization_percent_max', 0):.1f}%, "
              f"Avg: {gpu_agg.get('utilization_percent_avg', 0):.1f}%")
    
    if cpu_metrics:
        print("\n=== CPU Metrics (Aggregated) ===")
        cpu_agg = aggregated_data["aggregated_cpu_metrics"]
        print(f"CPU Usage - Min: {cpu_agg.get('usage_percent_min', 0):.1f}%, "
              f"Max: {cpu_agg.get('usage_percent_max', 0):.1f}%, "
              f"Avg: {cpu_agg.get('usage_percent_avg', 0):.1f}%")
        print(f"Memory Usage - Min: {cpu_agg.get('memory_used_mb_min', 0):.1f}MB, "
              f"Max: {cpu_agg.get('memory_used_mb_max', 0):.1f}MB, "
              f"Avg: {cpu_agg.get('memory_used_mb_avg', 0):.1f}MB")


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="Aggregate profiling reports from multiple ranks")
    parser.add_argument("--reports_dir", type=str, required=True, 
                        help="Directory containing rank subdirectories with profiling reports")
    parser.add_argument("--output", type=str, default="aggregated_profiling_report.json",
                        help="Output path for aggregated report")
    
    args = parser.parse_args()
    
    aggregate_reports(args.reports_dir, args.output)


if __name__ == "__main__":
    main()