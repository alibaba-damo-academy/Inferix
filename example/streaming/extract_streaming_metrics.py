"""
Extract Streaming Performance Metrics from Profiling Reports

This script extracts key performance metrics from Inferix profiling reports
and formats them for documentation updates.

Usage:
    python example/streaming/extract_streaming_metrics.py \
        --profile_dir ./profiling_results \
        --output_file benchmark_results.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional


def find_latest_profiling_report(profile_dir: str) -> Optional[str]:
    """Find the most recent JSON profiling report.
    
    Args:
        profile_dir: Directory containing profiling reports
        
    Returns:
        Path to the latest JSON report, or None if not found
    """
    profile_path = Path(profile_dir)
    
    if not profile_path.exists():
        print(f"Error: Profile directory does not exist: {profile_dir}")
        return None
    
    # Look for JSON reports
    json_reports = list(profile_path.glob("*.json"))
    
    if not json_reports:
        print(f"Error: No JSON profiling reports found in {profile_dir}")
        return None
    
    # Return the most recent one
    latest_report = max(json_reports, key=lambda p: p.stat().st_mtime)
    return str(latest_report)


def extract_metrics_from_report(report_path: str) -> Dict[str, Any]:
    """Extract streaming-relevant metrics from profiling report.
    
    Args:
        report_path: Path to JSON profiling report
        
    Returns:
        Dictionary with extracted metrics
    """
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    metrics = {
        'timestamp': report.get('timestamp', ''),
        'system_info': {},
        'block_level': {},
        'segment_level': {},
        'overall': {}
    }
    
    # Extract system information
    if 'system_info' in report:
        sys_info = report['system_info']
        metrics['system_info'] = {
            'gpu_name': sys_info.get('gpu_name', 'Unknown GPU'),
            'gpu_memory_total': sys_info.get('gpu_memory_total_gb', 0),
            'cuda_version': sys_info.get('cuda_version', 'Unknown'),
            'pytorch_version': sys_info.get('pytorch_version', 'Unknown')
        }
    
    # Extract diffusion analysis (block-level metrics)
    if 'diffusion_analysis' in report:
        diff = report['diffusion_analysis']
        metrics['block_level'] = {
            'total_steps': diff.get('total_steps', 0),
            'avg_block_size': diff.get('avg_block_size', 0),
            'fps': diff.get('fps', 0),
            'bps': diff.get('bps', 0),
            'avg_step_time_ms': diff.get('avg_step_time_ms', 0)
        }
    
    # Extract block computation analysis
    if 'block_analysis' in report:
        block = report['block_analysis']
        metrics['block_level']['total_blocks'] = block.get('total_blocks', 0)
        metrics['block_level']['block_fps'] = block.get('fps', 0)
        metrics['block_level']['peak_memory_mb'] = block.get('peak_memory_usage_mb', 0)
        metrics['block_level']['avg_block_time_ms'] = block.get('avg_computation_time_ms', 0)
    
    # Extract session-level metrics
    if 'session' in report:
        session = report['session']
        total_duration = session.get('duration', 0)
        
        metrics['overall']['total_duration_s'] = total_duration
        
        # Calculate segment-level metrics if available
        if 'tags' in session:
            tags = session['tags']
            num_segments = tags.get('num_segments', 1)
            
            if num_segments > 0 and total_duration > 0:
                metrics['segment_level']['avg_segment_time_s'] = total_duration / num_segments
                metrics['segment_level']['num_segments'] = num_segments
    
    # Extract GPU metrics
    if 'gpu_metrics' in report:
        gpu = report['gpu_metrics']
        metrics['overall']['peak_memory_mb'] = gpu.get('peak_memory_mb', 0)
        metrics['overall']['avg_gpu_utilization'] = gpu.get('avg_utilization', 0)
        metrics['overall']['peak_temperature_c'] = gpu.get('peak_temperature', 0)
    
    # Calculate throughput if we have total frames and duration
    total_frames = metrics['block_level'].get('total_steps', 0) * metrics['block_level'].get('avg_block_size', 0)
    total_duration = metrics['overall'].get('total_duration_s', 0)
    
    if total_frames > 0 and total_duration > 0:
        metrics['overall']['throughput_fps'] = total_frames / total_duration
    
    return metrics


def format_metrics_for_readme(metrics: Dict[str, Any]) -> str:
    """Format extracted metrics for README documentation.
    
    Args:
        metrics: Extracted metrics dictionary
        
    Returns:
        Formatted markdown string
    """
    md = []
    md.append("### Benchmark Results\n")
    md.append(f"**GPU**: {metrics['system_info'].get('gpu_name', 'Unknown')}\n")
    md.append(f"**VRAM**: {metrics['system_info'].get('gpu_memory_total', 0):.0f} GB\n")
    md.append("")
    
    md.append("**Block-level Performance:**")
    block = metrics['block_level']
    md.append(f"- Generation: {block.get('avg_step_time_ms', 0):.1f} ms per diffusion step")
    md.append(f"- Block Computation: {block.get('avg_block_time_ms', 0):.1f} ms per block")
    md.append(f"- Block FPS: {block.get('block_fps', 0):.2f}")
    md.append(f"- Blocks Per Second: {block.get('bps', 0):.2f}")
    md.append("")
    
    if metrics['segment_level']:
        md.append("**Segment-level Performance:**")
        segment = metrics['segment_level']
        md.append(f"- Time per segment: {segment.get('avg_segment_time_s', 0):.2f} s")
        md.append(f"- Total segments tested: {segment.get('num_segments', 0)}")
        md.append("")
    
    md.append("**Overall Performance:**")
    overall = metrics['overall']
    md.append(f"- Throughput: {overall.get('throughput_fps', 0):.2f} FPS")
    md.append(f"- Peak memory: {overall.get('peak_memory_mb', 0):.0f} MB")
    md.append(f"- GPU utilization: {overall.get('avg_gpu_utilization', 0):.1f}%")
    
    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(description='Extract streaming metrics from profiling reports')
    
    parser.add_argument('--profile_dir', type=str, required=True,
                       help='Directory containing profiling reports')
    parser.add_argument('--output_file', type=str, default='benchmark_results.json',
                       help='Output JSON file path')
    parser.add_argument('--print_markdown', action='store_true',
                       help='Print formatted markdown output')
    
    args = parser.parse_args()
    
    # Find latest report
    print(f"Searching for profiling reports in: {args.profile_dir}")
    report_path = find_latest_profiling_report(args.profile_dir)
    
    if not report_path:
        return 1
    
    print(f"Found profiling report: {report_path}")
    
    # Extract metrics
    print("Extracting streaming metrics...")
    metrics = extract_metrics_from_report(report_path)
    
    # Save to JSON
    with open(args.output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✅ Metrics saved to: {args.output_file}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("Performance Summary")
    print("=" * 70)
    print(f"GPU: {metrics['system_info'].get('gpu_name', 'Unknown')}")
    print(f"Block FPS: {metrics['block_level'].get('block_fps', 0):.2f}")
    print(f"Throughput: {metrics['overall'].get('throughput_fps', 0):.2f} FPS")
    print(f"Peak Memory: {metrics['overall'].get('peak_memory_mb', 0):.0f} MB")
    
    # Print markdown if requested
    if args.print_markdown:
        print("\n" + "=" * 70)
        print("Markdown for README.md")
        print("=" * 70)
        print(format_metrics_for_readme(metrics))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
