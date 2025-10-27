import json
import os
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

from .config import ProfilingConfig
from .monitors import MetricSnapshot


class ProfilingReporter:
    """Generate comprehensive profiling reports in multiple formats."""
    
    def __init__(self, config: ProfilingConfig):
        self.config = config
        
        # Set up output directory
        if config.output_dir:
            self.output_dir = config.output_dir
        else:
            self.output_dir = os.path.join(os.getcwd(), "profiling_reports")
        
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_report(self, session, gpu_data: List[MetricSnapshot], 
                       cpu_data: List[MetricSnapshot]) -> Optional[str]:
        """Generate profiling report(s) based on configuration.
        
        Args:
            session: ProfilingSession object
            gpu_data: List of GPU metric snapshots
            cpu_data: List of CPU metric snapshots
            
        Returns:
            Path to the generated report(s) or None if generation failed
        """
        try:
            # Prepare report data
            report_data = self._prepare_report_data(session, gpu_data, cpu_data)
            
            # Generate reports based on format configuration
            report_paths = []
            
            if self.config.report_format in ["json", "both"]:
                json_path = self._generate_json_report(session.session_id, report_data)
                if json_path:
                    report_paths.append(json_path)
            
            if self.config.report_format in ["html", "both"]:
                html_path = self._generate_html_report(session.session_id, report_data)
                if html_path:
                    report_paths.append(html_path)
            
            # Save raw data if requested
            if self.config.save_raw_data:
                raw_path = self._save_raw_data(session.session_id, gpu_data, cpu_data)
                if raw_path:
                    report_paths.append(raw_path)
            
            return ", ".join(report_paths) if report_paths else None
            
        except Exception as e:
            print(f"Error generating profiling report: {e}")
            return None
    
    def _prepare_report_data(self, session, gpu_data: List[MetricSnapshot], 
                           cpu_data: List[MetricSnapshot]) -> Dict[str, Any]:
        """Prepare comprehensive report data structure."""
        
        # Session metadata
        report_data = {
            "session_info": {
                "id": session.session_id,
                "start_time": session.start_time,
                "end_time": session.end_time,
                "duration": session.duration,
                "tags": session.tags,
                "generated_at": time.time()
            },
            "summary": {},
            "gpu_metrics": {},
            "cpu_metrics": {},
            "stage_analysis": {},
            "events": session.custom_events,
            "recommendations": []
        }
        
        # GPU analysis
        if gpu_data:
            report_data["gpu_metrics"] = self._analyze_gpu_metrics(gpu_data)
            report_data["summary"]["gpu"] = self._summarize_gpu_metrics(gpu_data)
        
        # CPU analysis
        if cpu_data:
            report_data["cpu_metrics"] = self._analyze_cpu_metrics(cpu_data)
            report_data["summary"]["cpu"] = self._summarize_cpu_metrics(cpu_data)
        
        # Stage analysis
        if session.stage_timings:
            report_data["stage_analysis"] = self._analyze_stage_timings(session.stage_timings)
        
        # Enhanced analysis
        if hasattr(session, 'streaming_events') and session.streaming_events:
            report_data["streaming_analysis"] = self._analyze_streaming_metrics(session.streaming_events, session.dropped_frames)
        
        if hasattr(session, 'layer_timings') and session.layer_timings:
            report_data["computation_analysis"] = self._analyze_computation_metrics(session.layer_timings)
        
        if hasattr(session, 'memory_events') and session.memory_events:
            report_data["memory_analysis"] = self._analyze_memory_metrics(session.memory_events)
        
        if hasattr(session, 'cache_stats') and session.cache_stats:
            report_data["cache_analysis"] = self._analyze_cache_metrics(session.cache_stats)
        
        # Diffusion model analysis
        if hasattr(session, 'custom_events'):
            # Look for diffusion-specific events
            diffusion_steps = [event for event in session.custom_events if event.get('name') == 'diffusion_step']
            model_parameters = [event for event in session.custom_events if event.get('name') == 'model_parameters']
            block_computations = [event for event in session.custom_events if event.get('name') == 'block_computation']
            
            if diffusion_steps:
                report_data["diffusion_analysis"] = self._analyze_diffusion_metrics(diffusion_steps)
            
            if model_parameters:
                report_data["model_analysis"] = self._analyze_model_metrics(model_parameters)
            
            if block_computations:
                report_data["block_analysis"] = self._analyze_block_metrics(block_computations)
        
        # Generate recommendations
        report_data["recommendations"] = self._generate_recommendations(report_data)
        
        return report_data
    
    def _analyze_gpu_metrics(self, gpu_data: List[MetricSnapshot]) -> Dict[str, Any]:
        """Analyze GPU metrics and extract insights."""
        analysis = {
            "total_samples": len(gpu_data),
            "timeline": [],
            "memory_analysis": {},
            "utilization_analysis": {},
            "temperature_analysis": {},
            "power_analysis": {}
        }
        
        if not gpu_data:
            return analysis
        
        # Extract timeline data
        memory_usage = []
        gpu_utilization = []
        temperatures = []
        power_consumption = []
        
        for snapshot in gpu_data:
            metrics = snapshot.metrics
            analysis["timeline"].append({
                "timestamp": snapshot.timestamp,
                "memory_used_mb": metrics.get("total_gpu_memory_used_mb", 0),
                "memory_total_mb": metrics.get("total_gpu_memory_total_mb", 0),
                "utilization": metrics.get("max_gpu_utilization", 0),
                "temperature": metrics.get("max_gpu_temperature", 0),
                "power": metrics.get("total_gpu_power_watts", 0)
            })
            
            memory_usage.append(metrics.get("total_gpu_memory_used_mb", 0))
            gpu_utilization.append(metrics.get("max_gpu_utilization", 0))
            temperatures.append(metrics.get("max_gpu_temperature", 0))
            power_consumption.append(metrics.get("total_gpu_power_watts", 0))
        
        # Memory analysis
        if memory_usage:
            analysis["memory_analysis"] = {
                "peak_usage_mb": max(memory_usage),
                "avg_usage_mb": sum(memory_usage) / len(memory_usage),
                "min_usage_mb": min(memory_usage),
                "total_memory_mb": gpu_data[0].metrics.get("total_gpu_memory_total_mb", 0),
                "peak_utilization_percent": (max(memory_usage) / max(gpu_data[0].metrics.get("total_gpu_memory_total_mb", 1), 1)) * 100
            }
        
        # Utilization analysis
        if gpu_utilization:
            analysis["utilization_analysis"] = {
                "peak_utilization": max(gpu_utilization),
                "avg_utilization": sum(gpu_utilization) / len(gpu_utilization),
                "min_utilization": min(gpu_utilization),
                "utilization_efficiency": self._calculate_efficiency_score(gpu_utilization)
            }
        
        # Temperature analysis
        if temperatures and any(t > 0 for t in temperatures):
            analysis["temperature_analysis"] = {
                "peak_temperature": max(temperatures),
                "avg_temperature": sum(temperatures) / len(temperatures),
                "min_temperature": min(temperatures),
                "thermal_throttling_risk": max(temperatures) > 85  # Typical thermal throttling threshold
            }
        
        # Power analysis
        if power_consumption and any(p > 0 for p in power_consumption):
            analysis["power_analysis"] = {
                "peak_power_watts": max(power_consumption),
                "avg_power_watts": sum(power_consumption) / len(power_consumption),
                "total_energy_wh": (sum(power_consumption) * len(gpu_data) * 0.5) / 3600  # Approximate energy consumption
            }
        
        return analysis
    
    def _analyze_cpu_metrics(self, cpu_data: List[MetricSnapshot]) -> Dict[str, Any]:
        """Analyze CPU metrics and extract insights."""
        analysis = {
            "total_samples": len(cpu_data),
            "timeline": [],
            "cpu_analysis": {},
            "memory_analysis": {}
        }
        
        if not cpu_data:
            return analysis
        
        # Extract timeline data
        cpu_usage = []
        memory_usage = []
        
        for snapshot in cpu_data:
            metrics = snapshot.metrics
            analysis["timeline"].append({
                "timestamp": snapshot.timestamp,
                "cpu_usage": metrics.get("cpu_usage_percent", 0),
                "memory_used_mb": metrics.get("memory_used_mb", 0),
                "memory_total_mb": metrics.get("memory_total_mb", 0)
            })
            
            cpu_usage.append(metrics.get("cpu_usage_percent", 0))
            memory_usage.append(metrics.get("memory_used_mb", 0))
        
        # CPU analysis
        if cpu_usage:
            analysis["cpu_analysis"] = {
                "peak_usage": max(cpu_usage),
                "avg_usage": sum(cpu_usage) / len(cpu_usage),
                "min_usage": min(cpu_usage)
            }
        
        # Memory analysis
        if memory_usage:
            total_memory = cpu_data[0].metrics.get("memory_total_mb", 0)
            analysis["memory_analysis"] = {
                "peak_usage_mb": max(memory_usage),
                "avg_usage_mb": sum(memory_usage) / len(memory_usage),
                "min_usage_mb": min(memory_usage),
                "total_memory_mb": total_memory,
                "peak_utilization_percent": (max(memory_usage) / max(total_memory, 1)) * 100
            }
        
        return analysis
    
    def _analyze_stage_timings(self, stage_timings: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze pipeline stage timings."""
        analysis = {
            "total_stages": len(stage_timings),
            "total_time": 0.0,
            "stages": {},
            "bottlenecks": []
        }
        
        # Analyze each stage
        for stage_name, timing_data in stage_timings.items():
            total_time = timing_data.get("total_time", 0.0)
            call_count = timing_data.get("call_count", 1)
            
            stage_analysis = {
                "total_time": total_time,
                "call_count": call_count,
                "avg_time": total_time / max(call_count, 1),
                "min_time": timing_data.get("min_time", 0.0),
                "max_time": timing_data.get("max_time", 0.0),
                "percentage_of_total": 0.0  # Will be calculated below
            }
            
            analysis["stages"][stage_name] = stage_analysis
            analysis["total_time"] += total_time
        
        # Calculate percentages and identify bottlenecks
        if analysis["total_time"] > 0:
            for stage_name, stage_data in analysis["stages"].items():
                percentage = (stage_data["total_time"] / analysis["total_time"]) * 100
                stage_data["percentage_of_total"] = percentage
                
                # Consider stages taking >30% of total time as bottlenecks
                if percentage > 30:
                    analysis["bottlenecks"].append({
                        "stage": stage_name,
                        "percentage": percentage,
                        "total_time": stage_data["total_time"]
                    })
        
        # Sort bottlenecks by impact
        analysis["bottlenecks"].sort(key=lambda x: x["percentage"], reverse=True)
        
        return analysis
    
    def _analyze_diffusion_metrics(self, diffusion_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze diffusion model metrics."""
        if not diffusion_events:
            return {}
        
        analysis = {
            "total_steps": len(diffusion_events),
            "steps": []
        }
        
        # Extract step data
        computation_times = []
        timesteps = []
        block_sizes = []
        guidance_scales = []
        
        for event in diffusion_events:
            data = event.get("data", {})
            analysis["steps"].append(data)
            
            computation_times.append(data.get("computation_time_ms", 0))
            timesteps.append(data.get("timestep", 0))
            block_sizes.append(data.get("block_size", 0))
            if data.get("guidance_scale") is not None:
                guidance_scales.append(data.get("guidance_scale"))
        
        # Calculate statistics
        if computation_times:
            analysis["total_computation_time_ms"] = sum(computation_times)
            analysis["avg_computation_time_ms"] = sum(computation_times) / len(computation_times)
            analysis["min_computation_time_ms"] = min(computation_times)
            analysis["max_computation_time_ms"] = max(computation_times)
        
        if timesteps:
            analysis["avg_timestep"] = sum(timesteps) / len(timesteps)
            analysis["min_timestep"] = min(timesteps)
            analysis["max_timestep"] = max(timesteps)
        
        if block_sizes:
            analysis["avg_block_size"] = sum(block_sizes) / len(block_sizes)
            analysis["min_block_size"] = min(block_sizes)
            analysis["max_block_size"] = max(block_sizes)
            analysis["total_frames"] = sum(block_sizes)
        
        # Calculate FPS and BPS
        total_time_seconds = analysis.get("total_computation_time_ms", 0) / 1000.0
        if total_time_seconds > 0:
            analysis["fps"] = analysis.get("total_frames", 0) / total_time_seconds
            analysis["bps"] = analysis["total_steps"] / total_time_seconds
        else:
            analysis["fps"] = 0
            analysis["bps"] = 0
        
        return analysis
    
    def _analyze_model_metrics(self, model_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze model parameter metrics.
        
        Note: This analysis focuses on parameter counts but does not consider
        actual computational load or memory usage patterns during execution.
        For performance optimization, focus on the diffusion analysis and
        block computation analysis which provide more actionable insights.
        """
        if not model_events:
            return {}
        
        analysis = {
            "models": {},
            "total_parameters": 0
        }
        
        # Extract model data
        for event in model_events:
            data = event.get("data", {})
            model_name = data.get("model_name")
            parameters_count = data.get("parameters_count", 0)
            model_type = data.get("model_type")
            
            if model_name:
                analysis["models"][model_name] = {
                    "parameters_count": parameters_count,
                    "model_type": model_type
                }
                analysis["total_parameters"] += parameters_count
        
        # Format readable parameter count
        if analysis["total_parameters"] >= 1e9:
            analysis["total_parameters_readable"] = f"{analysis['total_parameters'] / 1e9:.1f}B"
        elif analysis["total_parameters"] >= 1e6:
            analysis["total_parameters_readable"] = f"{analysis['total_parameters'] / 1e6:.1f}M"
        else:
            analysis["total_parameters_readable"] = f"{analysis['total_parameters']:,}"
        
        return analysis
    
    def _analyze_block_metrics(self, block_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze block computation metrics."""
        if not block_events:
            return {}
        
        analysis = {
            "total_blocks": len(block_events),
            "blocks": []
        }
        
        # Extract block data
        computation_times = []
        block_sizes = []
        memory_usages = []
        
        for event in block_events:
            data = event.get("data", {})
            analysis["blocks"].append(data)
            
            computation_times.append(data.get("computation_time_ms", 0))
            block_sizes.append(data.get("block_size", 0))
            memory_usages.append(data.get("memory_usage_mb", 0))
        
        # Calculate statistics
        if computation_times:
            analysis["total_computation_time_ms"] = sum(computation_times)
            analysis["avg_computation_time_ms"] = sum(computation_times) / len(computation_times)
            analysis["min_computation_time_ms"] = min(computation_times)
            analysis["max_computation_time_ms"] = max(computation_times)
        
        if block_sizes:
            analysis["avg_block_size"] = sum(block_sizes) / len(block_sizes)
            analysis["min_block_size"] = min(block_sizes)
            analysis["max_block_size"] = max(block_sizes)
            analysis["total_frames"] = sum(block_sizes)
        
        if memory_usages:
            analysis["avg_memory_usage_mb"] = sum(memory_usages) / len(memory_usages)
            analysis["peak_memory_usage_mb"] = max(memory_usages)
            analysis["min_memory_usage_mb"] = min(memory_usages)
        
        # Calculate FPS and BPS
        total_time_seconds = analysis.get("total_computation_time_ms", 0) / 1000.0
        if total_time_seconds > 0:
            analysis["fps"] = analysis.get("total_frames", 0) / total_time_seconds
            analysis["bps"] = analysis["total_blocks"] / total_time_seconds
        else:
            analysis["fps"] = 0
            analysis["bps"] = 0
        
        return analysis
    
    def _generate_recommendations(self, report_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # GPU recommendations
        gpu_summary = report_data.get("summary", {}).get("gpu", {})
        if gpu_summary:
            memory_utilization = gpu_summary.get("peak_memory_utilization", 0)
            gpu_utilization = gpu_summary.get("avg_gpu_utilization", 0)
            
            if memory_utilization > 90:
                recommendations.append({
                    "category": "GPU Memory",
                    "priority": "High",
                    "issue": f"Peak GPU memory utilization is {memory_utilization:.1f}%",
                    "suggestion": "Consider reducing batch size, enabling gradient checkpointing, or using mixed precision training to reduce memory usage."
                })
            
            if gpu_utilization < 50:
                recommendations.append({
                    "category": "GPU Utilization",
                    "priority": "Medium",
                    "issue": f"Average GPU utilization is only {gpu_utilization:.1f}%",
                    "suggestion": "Consider increasing batch size, optimizing data loading, or reducing CPU bottlenecks to better utilize GPU."
                })
        
        # Stage timing recommendations
        stage_analysis = report_data.get("stage_analysis", {})
        bottlenecks = stage_analysis.get("bottlenecks", [])
        
        for bottleneck in bottlenecks[:3]:  # Top 3 bottlenecks
            # Skip the generic "run_inference" bottleneck recommendation as it's expected to be high
            if bottleneck['stage'] != 'run_inference':
                recommendations.append({
                    "category": "Performance Bottleneck",
                    "priority": "High",
                    "issue": f"Stage '{bottleneck['stage']}' takes {bottleneck['percentage']:.1f}% of total time",
                    "suggestion": f"Focus optimization efforts on the '{bottleneck['stage']}' stage as it's a major bottleneck."
                })
        
        # Temperature recommendations
        gpu_metrics = report_data.get("gpu_metrics", {})
        temp_analysis = gpu_metrics.get("temperature_analysis", {})
        if temp_analysis.get("thermal_throttling_risk", False):
            recommendations.append({
                "category": "Thermal Management",
                "priority": "High",
                "issue": f"Peak GPU temperature is {temp_analysis.get('peak_temperature', 0):.1f}°C",
                "suggestion": "GPU temperature is high and may cause thermal throttling. Improve cooling or reduce GPU load."
            })
        
        # Streaming recommendations
        streaming_analysis = report_data.get("streaming_analysis", {})
        if streaming_analysis.get("drop_rate_percent", 0) > 5:  # >5% drop rate
            recommendations.append({
                "category": "Streaming Performance",
                "priority": "High",
                "issue": f"High frame drop rate: {streaming_analysis.get('drop_rate_percent', 0):.1f}%",
                "suggestion": "Reduce encoding complexity or improve network bandwidth to maintain streaming quality."
            })
        
        # Computation recommendations
        computation_analysis = report_data.get("computation_analysis", {})
        bottlenecks = computation_analysis.get("bottlenecks", [])
        for bottleneck in bottlenecks[:3]:  # Top 3 bottlenecks
            recommendations.append({
                "category": "Computation Bottleneck",
                "priority": "High",
                "issue": f"Layer '{bottleneck['layer']}' takes {bottleneck['percentage']:.1f}% of total computation time",
                "suggestion": f"Consider optimizing '{bottleneck['layer']}' layer computation or redistributing workload."
            })
        
        # Diffusion model recommendations
        diffusion_analysis = report_data.get("diffusion_analysis", {})
        model_analysis = report_data.get("model_analysis", {})
        block_analysis = report_data.get("block_analysis", {})
        
        # Diffusion steps recommendations
        if diffusion_analysis and diffusion_analysis.get("total_steps", 0) > 30:
            recommendations.append({
                "category": "Diffusion Steps",
                "priority": "High",
                "issue": f"Using {diffusion_analysis.get('total_steps', 0)} diffusion steps",
                "suggestion": "Consider reducing steps to 20-30 to improve generation speed."
            })
        
        # Block size recommendations
        if block_analysis and block_analysis.get("avg_block_size", 0) > 8:
            recommendations.append({
                "category": "Block Size",
                "priority": "Medium",
                "issue": f"Average block size is {block_analysis.get('avg_block_size', 0):.1f} frames",
                "suggestion": "Consider reducing block size to improve latency and memory usage. Smaller blocks enable better streaming performance."
            })
        
        # FPS recommendations
        if diffusion_analysis and diffusion_analysis.get("fps", 0) < 10:
            recommendations.append({
                "category": "Generation Speed",
                "priority": "High",
                "issue": f"Current FPS is {diffusion_analysis.get('fps', 0):.1f}",
                "suggestion": f"Current generation speed is below real-time requirements. Consider optimizing the model or reducing the computational load to achieve at least 16 FPS for smooth video generation."
            })
        
        # BPS recommendations
        if diffusion_analysis and diffusion_analysis.get("bps", 0) < 2:
            recommendations.append({
                "category": "Block Processing",
                "priority": "Medium",
                "issue": f"Current BPS is {diffusion_analysis.get('bps', 0):.1f}",
                "suggestion": f"Current block processing speed is relatively slow. Consider optimizing block processing or using parallel processing to improve throughput."
            })
        
        # Multi-GPU scaling recommendation with accurate parallel information
        session_tags = report_data.get("session_info", {}).get("tags", {})
        world_size = session_tags.get("world_size", 1)
        ulysses_size = session_tags.get("ulysses_size", 1)
        ring_size = session_tags.get("ring_size", 1)
        
        # Add first-block delay analysis if available
        first_block_delay = self._calculate_first_block_delay(report_data)
        if first_block_delay:
            recommendations.append({
                "category": "Latency Optimization",
                "priority": "Medium",
                "issue": f"First-block delay is {first_block_delay:.2f} seconds",
                "suggestion": "The time to generate the first video block is significant. Consider optimizing initialization or using smaller block sizes to reduce initial latency."
            })
        
        if world_size > 1:
            # More accurate parallel strategy recommendations
            parallel_strategy = []
            if ulysses_size > 1:
                parallel_strategy.append(f"Ulysses ({ulysses_size} GPUs)")
            if ring_size > 1:
                parallel_strategy.append(f"Ring ({ring_size} GPUs)")
            
            strategy_str = " and ".join(parallel_strategy) if parallel_strategy else "Data Parallel"
            
            recommendations.append({
                "category": "Multi-GPU Scaling",
                "priority": "Medium",
                "issue": f"Using {world_size} GPUs with {strategy_str} strategy",
                "suggestion": "You're already using multi-GPU parallelization. For better scaling, consider: "
                            "1) Experimenting with different ulysses/ring combinations based on your GPU interconnect bandwidth; "
                            "2) Exploring advanced parallelization strategies like block-parallel or timestep-parallel; "
                            "3) Note that scaling is not perfectly linear due to communication overhead."
            })
            
            # Advanced parallelization suggestions
            recommendations.append({
                "category": "Advanced Parallelization",
                "priority": "Low",
                "issue": "Opportunity for more sophisticated parallel strategies",
                "suggestion": "Consider implementing more advanced parallelization: "
                            "1) Block-parallel: Process different video blocks on different GPUs; "
                            "2) Timestep-parallel: Pipeline diffusion timesteps across GPUs; "
                            "3) Hybrid approaches: Combine block and timestep parallelization for optimal throughput."
            })
        else:
            recommendations.append({
                "category": "Multi-GPU Scaling",
                "priority": "Medium",
                "issue": "Using single GPU",
                "suggestion": "Consider using multiple GPUs to accelerate generation. "
                            "You can use Ulysses parallel (for attention computation) and Ring parallel (for sequence processing) "
                            "in various combinations depending on your GPU interconnect bandwidth."
            })
        
        return recommendations

    def _calculate_first_block_delay(self, report_data: Dict[str, Any]) -> Optional[float]:
        """Calculate the first-block delay - equivalent to first-token delay in LLMs."""
        # Look for block computation events to determine first block generation time
        block_analysis = report_data.get("block_analysis", {})
        blocks = block_analysis.get("blocks", [])
        
        if blocks:
            # First block is typically the first in the list
            first_block = blocks[0] if isinstance(blocks, list) else {}
            return first_block.get("computation_time_ms", 0) / 1000.0  # Convert ms to seconds
        
        # Alternative: Look for stage timings for the first inference block
        stage_analysis = report_data.get("stage_analysis", {})
        stages = stage_analysis.get("stages", {})
        
        # Look for stages related to block processing
        for stage_name, stage_data in stages.items():
            if "block" in stage_name.lower() or "diffusion" in stage_name.lower():
                # Assume the first call is for the first block
                if stage_data.get("call_count", 0) > 0:
                    return stage_data.get("avg_time", 0)
        
        return None
    
    def _analyze_streaming_metrics(self, streaming_events: List[Dict[str, Any]], dropped_frames: int) -> Dict[str, Any]:
        """Analyze streaming performance metrics."""
        if not streaming_events:
            return {}
        
        analysis = {
            "total_streaming_events": len(streaming_events),
            "total_frames_streamed": 0,
            "total_dropped_frames": dropped_frames,
            "current_fps": 0.0,
            "avg_encoding_time_ms": 0.0,
            "avg_network_latency_ms": 0.0,
            "drop_rate_percent": 0.0
        }
        
        # Calculate totals
        total_frames = sum(event.get("batch_size", 0) for event in streaming_events)
        analysis["total_frames_streamed"] = total_frames
        
        # Calculate averages
        encoding_times = [event.get("encoding_time_ms", 0) for event in streaming_events]
        network_latencies = [event.get("network_latency_ms", 0) for event in streaming_events]
        
        if encoding_times:
            analysis["avg_encoding_time_ms"] = sum(encoding_times) / len(encoding_times)
        
        if network_latencies:
            analysis["avg_network_latency_ms"] = sum(network_latencies) / len(network_latencies)
        
        # Calculate FPS based on recent events
        if len(streaming_events) >= 2:
            recent_events = streaming_events[-10:]  # Last 10 events
            time_span = recent_events[-1].get("timestamp", 0) - recent_events[0].get("timestamp", 0)
            frame_span = sum(event.get("batch_size", 0) for event in recent_events)
            
            if time_span > 0:
                analysis["current_fps"] = frame_span / time_span
        
        # Drop rate
        total_frames_with_drops = total_frames + dropped_frames
        if total_frames_with_drops > 0:
            analysis["drop_rate_percent"] = (dropped_frames / total_frames_with_drops) * 100
        
        return analysis
    
    def _analyze_computation_metrics(self, layer_timings: Dict[str, List[float]]) -> Dict[str, Any]:
        """Analyze computation performance metrics."""
        if not layer_timings:
            return {}
        
        analysis = {
            "layer_performance": {},
            "bottlenecks": []
        }
        
        # Layer performance analysis
        total_computation_time = 0
        layer_total_times = {}
        
        for layer_name, timings in layer_timings.items():
            if timings:
                total_time = sum(timings)
                layer_total_times[layer_name] = total_time
                total_computation_time += total_time
                
                analysis["layer_performance"][layer_name] = {
                    "avg_time_ms": sum(timings) / len(timings),
                    "min_time_ms": min(timings),
                    "max_time_ms": max(timings),
                    "total_time_ms": total_time,
                    "call_count": len(timings)
                }
        
        # Identify bottlenecks (>30% of total time)
        if total_computation_time > 0:
            for layer_name, total_time in layer_total_times.items():
                percentage = (total_time / total_computation_time) * 100
                if percentage > 30:
                    analysis["bottlenecks"].append({
                        "layer": layer_name,
                        "percentage": percentage,
                        "total_time_ms": total_time
                    })
            
            # Sort bottlenecks by impact
            analysis["bottlenecks"].sort(key=lambda x: x["percentage"], reverse=True)
        
        return analysis
    
    def _analyze_memory_metrics(self, memory_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze memory usage patterns."""
        if not memory_events:
            return {}
        
        analysis = {
            "allocation_patterns": {},
            "fragmentation_analysis": {}
        }
        
        # Group by allocator
        allocators = {}
        for event in memory_events:
            allocator = event.get("allocator", "unknown")
            if allocator not in allocators:
                allocators[allocator] = []
            allocators[allocator].append(event)
        
        # Analyze each allocator
        for allocator, events in allocators.items():
            sizes = [event.get("size_bytes", 0) for event in events]
            analysis["allocation_patterns"][allocator] = {
                "total_allocations": len([e for e in events if e.get("event_type") == "alloc"]),
                "total_deallocations": len([e for e in events if e.get("event_type") == "dealloc"]),
                "total_bytes_allocated": sum(sizes),
                "avg_allocation_size_bytes": sum(sizes) / len(sizes) if sizes else 0,
                "max_allocation_size_bytes": max(sizes) if sizes else 0,
                "min_allocation_size_bytes": min(sizes) if sizes else 0
            }
        
        # Fragmentation analysis (simplified)
        alloc_events = [e for e in memory_events if e.get("event_type") == "alloc"]
        if alloc_events:
            # Simple fragmentation metric: ratio of small allocations
            small_allocations = [e for e in alloc_events if e.get("size_bytes", 0) < 1024 * 1024]  # < 1MB
            analysis["fragmentation_analysis"] = {
                "small_allocation_ratio": len(small_allocations) / len(alloc_events) if alloc_events else 0,
                "total_allocations": len(alloc_events)
            }
        
        return analysis
    
    def _analyze_cache_metrics(self, cache_stats: Dict[str, int]) -> Dict[str, Any]:
        """Analyze cache performance metrics."""
        if not cache_stats or cache_stats.get("total_requests", 0) == 0:
            return {}
        
        total_requests = cache_stats.get("total_requests", 1)
        hits = cache_stats.get("hits", 0)
        misses = cache_stats.get("misses", 0)
        evictions = cache_stats.get("evictions", 0)
        
        analysis = {
            "cache_performance": {
                "hits": hits,
                "misses": misses,
                "evictions": evictions,
                "total_requests": total_requests
            },
            "cache_efficiency": {
                "hit_rate": hits / total_requests,
                "miss_rate": misses / total_requests,
                "eviction_rate": evictions / total_requests
            }
        }
        
        return analysis
    
    def _calculate_efficiency_score(self, utilization_data: List[float]) -> float:
        """Calculate a simple efficiency score based on utilization consistency."""
        if not utilization_data:
            return 0.0
        
        avg_util = sum(utilization_data) / len(utilization_data)
        variance = sum((x - avg_util) ** 2 for x in utilization_data) / len(utilization_data)
        
        # Score based on average utilization and consistency (lower variance is better)
        consistency_score = max(0, 100 - variance)
        return min(100.0, (avg_util + consistency_score) / 2)
    
    def _summarize_gpu_metrics(self, gpu_data: List[MetricSnapshot]) -> Dict[str, Any]:
        """Create GPU metrics summary."""
        if not gpu_data:
            return {}
        
        memory_values = [s.metrics.get("total_gpu_memory_used_mb", 0) for s in gpu_data]
        utilization_values = [s.metrics.get("max_gpu_utilization", 0) for s in gpu_data]
        
        return {
            "peak_memory_usage_mb": max(memory_values) if memory_values else 0,
            "avg_memory_usage_mb": sum(memory_values) / len(memory_values) if memory_values else 0,
            "peak_memory_utilization": (max(memory_values) / max(gpu_data[0].metrics.get("total_gpu_memory_total_mb", 1), 1)) * 100 if memory_values else 0,
            "avg_gpu_utilization": sum(utilization_values) / len(utilization_values) if utilization_values else 0,
            "peak_gpu_utilization": max(utilization_values) if utilization_values else 0
        }
    
    def _summarize_cpu_metrics(self, cpu_data: List[MetricSnapshot]) -> Dict[str, Any]:
        """Create CPU metrics summary."""
        if not cpu_data:
            return {}
        
        cpu_values = [s.metrics.get("cpu_usage_percent", 0) for s in cpu_data]
        memory_values = [s.metrics.get("memory_used_mb", 0) for s in cpu_data]
        
        return {
            "avg_cpu_usage": sum(cpu_values) / len(cpu_values) if cpu_values else 0,
            "peak_cpu_usage": max(cpu_values) if cpu_values else 0,
            "avg_memory_usage_mb": sum(memory_values) / len(memory_values) if memory_values else 0,
            "peak_memory_usage_mb": max(memory_values) if memory_values else 0
        }
    
    def _generate_json_report(self, session_id: str, report_data: Dict[str, Any]) -> Optional[str]:
        """Generate JSON format report."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"profiling_report_{session_id}_{timestamp}.json"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            return filepath
        except Exception as e:
            print(f"Error generating JSON report: {e}")
            return None
    
    def _generate_html_report(self, session_id: str, report_data: Dict[str, Any]) -> Optional[str]:
        """Generate HTML format report."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"profiling_report_{session_id}_{timestamp}.html"
            filepath = os.path.join(self.output_dir, filename)
            
            html_content = self._create_html_template(report_data)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return filepath
        except Exception as e:
            print(f"Error generating HTML report: {e}")
            return None
    
    def _create_html_template(self, data: Dict[str, Any]) -> str:
        """Create HTML report template."""
        session = data.get("session_info", {})
        summary = data.get("summary", {})
        gpu_metrics = data.get("gpu_metrics", {})
        cpu_metrics = data.get("cpu_metrics", {})
        stage_analysis = data.get("stage_analysis", {})
        recommendations = data.get("recommendations", [])
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Inferix Profiling Report - {session.get('id', 'Unknown')}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; margin-bottom: 30px; padding-bottom: 20px; border-bottom: 2px solid #e0e0e0; }}
        .section {{ margin-bottom: 30px; }}
        .section h2 {{ color: #333; border-left: 4px solid #007acc; padding-left: 15px; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin-bottom: 20px; }}
        .metric-card {{ background: #f8f9fa; padding: 15px; border-radius: 6px; border-left: 3px solid #007acc; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #007acc; }}
        .metric-label {{ color: #666; font-size: 14px; }}
        .recommendation {{ padding: 15px; margin-bottom: 10px; border-radius: 6px; border-left: 4px solid #28a745; background: #f8fff9; }}
        .recommendation.high {{ border-left-color: #dc3545; background: #fff5f5; }}
        .recommendation.medium {{ border-left-color: #ffc107; background: #fffdf5; }}
        .stage-table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        .stage-table th, .stage-table td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        .stage-table th {{ background: #f8f9fa; font-weight: 600; }}
        .timeline {{ font-family: monospace; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Inferix Profiling Report</h1>
            <p><strong>Session:</strong> {session.get('id', 'Unknown')} | 
               <strong>Duration:</strong> {session.get('duration', 0):.2f}s | 
               <strong>Generated:</strong> {datetime.fromtimestamp(session.get('generated_at', time.time())).strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
"""
        
        # Summary section
        if summary:
            html += """
        <div class="section">
            <h2>📊 Performance Summary</h2>
            <div class="metric-grid">
"""
            gpu_sum = summary.get("gpu", {})
            if gpu_sum:
                html += f"""
                <div class="metric-card">
                    <div class="metric-value">{gpu_sum.get('peak_gpu_utilization', 0):.1f}%</div>
                    <div class="metric-label">Peak GPU Utilization</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{gpu_sum.get('peak_memory_usage_mb', 0):.0f}MB</div>
                    <div class="metric-label">Peak GPU Memory</div>
                </div>
"""
            
            cpu_sum = summary.get("cpu", {})
            if cpu_sum:
                html += f"""
                <div class="metric-card">
                    <div class="metric-value">{cpu_sum.get('peak_cpu_usage', 0):.1f}%</div>
                    <div class="metric-label">Peak CPU Usage</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{cpu_sum.get('peak_memory_usage_mb', 0):.0f}MB</div>
                    <div class="metric-label">Peak System Memory</div>
                </div>
"""
            html += """
            </div>
        </div>
"""
        
        # Enhanced analysis sections
        
        # Diffusion analysis section
        diffusion_analysis = data.get("diffusion_analysis", {})
        if diffusion_analysis:
            html += """
        <div class="section">
            <h2>🌀 Diffusion Model Analysis</h2>
            <div class="metric-grid">
"""
            html += f"""
                <div class="metric-card">
                    <div class="metric-value">{diffusion_analysis.get('total_steps', 0)}</div>
                    <div class="metric-label">Diffusion Steps</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{diffusion_analysis.get('avg_block_size', 0):.1f}</div>
                    <div class="metric-label">Avg Block Size (frames)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{diffusion_analysis.get('avg_timestep', 0):.3f}</div>
                    <div class="metric-label">Avg Timestep</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{diffusion_analysis.get('fps', 0):.1f}</div>
                    <div class="metric-label">FPS (Frames/Second)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{diffusion_analysis.get('bps', 0):.1f}</div>
                    <div class="metric-label">BPS (Blocks/Second)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{diffusion_analysis.get('avg_computation_time_ms', 0):.1f}ms</div>
                    <div class="metric-label">Avg Step Time</div>
                </div>
"""
            html += """
            </div>
        </div>
"""
        
        # Model analysis section
        model_analysis = data.get("model_analysis", {})
        if model_analysis:
            html += """
        <div class="section">
            <h2>🧠 Model Analysis</h2>
            <p><strong>Note:</strong> Parameter counts do not directly correlate with computational load or memory usage during execution. 
            For performance optimization, focus on the diffusion analysis and block computation analysis which provide more actionable insights.</p>
            <div class="metric-grid">
"""
            html += f"""
                <div class="metric-card">
                    <div class="metric-value">{model_analysis.get('total_parameters_readable', '0')}</div>
                    <div class="metric-label">Total Parameters</div>
                </div>
"""
            html += """
            </div>
            <table class="stage-table">
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Parameters</th>
                        <th>Type</th>
                    </tr>
                </thead>
                <tbody>
"""
            for model_name, model_info in model_analysis.get("models", {}).items():
                html += f"""
                    <tr>
                        <td>{model_name}</td>
                        <td>{model_info.get('parameters_count', 0):,}</td>
                        <td>{model_info.get('model_type', 'unknown')}</td>
                    </tr>
"""
            html += """
                </tbody>
            </table>
        </div>
"""
        
        # Block analysis section
        block_analysis = data.get("block_analysis", {})
        if block_analysis:
            html += """
        <div class="section">
            <h2>📦 Block Computation Analysis</h2>
            <div class="metric-grid">
"""
            html += f"""
                <div class="metric-card">
                    <div class="metric-value">{block_analysis.get('total_blocks', 0)}</div>
                    <div class="metric-label">Total Blocks</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{block_analysis.get('avg_block_size', 0):.1f}</div>
                    <div class="metric-label">Avg Block Size</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{block_analysis.get('fps', 0):.1f}</div>
                    <div class="metric-label">FPS (Frames/Second)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{block_analysis.get('bps', 0):.1f}</div>
                    <div class="metric-label">BPS (Blocks/Second)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{block_analysis.get('avg_computation_time_ms', 0):.1f}ms</div>
                    <div class="metric-label">Avg Block Time</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{block_analysis.get('peak_memory_usage_mb', 0):.0f}MB</div>
                    <div class="metric-label">Peak Memory Usage</div>
                </div>
"""
            html += """
            </div>
        </div>
"""
        
        # Streaming analysis section
        streaming_analysis = data.get("streaming_analysis", {})
        if streaming_analysis:
            html += """
        <div class="section">
            <h2>📡 Streaming Performance Analysis</h2>
            <div class="metric-grid">
"""
            html += f"""
                <div class="metric-card">
                    <div class="metric-value">{streaming_analysis.get('current_fps', 0):.1f}</div>
                    <div class="metric-label">Current FPS</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{streaming_analysis.get('drop_rate_percent', 0):.1f}%</div>
                    <div class="metric-label">Drop Rate</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{streaming_analysis.get('avg_encoding_time_ms', 0):.1f}ms</div>
                    <div class="metric-label">Avg Encoding Time</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{streaming_analysis.get('total_frames_streamed', 0)}</div>
                    <div class="metric-label">Total Frames Streamed</div>
                </div>
"""
            html += """
            </div>
        </div>
"""
        
        # Computation analysis section
        computation_analysis = data.get("computation_analysis", {})
        layer_performance = computation_analysis.get("layer_performance", {})
        if layer_performance:
            html += """
        <div class="section">
            <h2>⚙️ Computation Performance Analysis</h2>
            <table class="stage-table">
                <thead>
                    <tr>
                        <th>Layer</th>
                        <th>Total Time (ms)</th>
                        <th>Avg Time (ms)</th>
                        <th>Call Count</th>
                    </tr>
                </thead>
                <tbody>
"""
            for layer_name, layer_data in layer_performance.items():
                html += f"""
                    <tr>
                        <td>{layer_name}</td>
                        <td>{layer_data.get('total_time_ms', 0):.1f}</td>
                        <td>{layer_data.get('avg_time_ms', 0):.1f}</td>
                        <td>{layer_data.get('call_count', 0)}</td>
                    </tr>
"""
            html += """
                </tbody>
            </table>
        </div>
"""
        
        # Cache analysis section
        cache_analysis = data.get("cache_analysis", {})
        cache_efficiency = cache_analysis.get("cache_efficiency", {})
        if cache_efficiency:
            html += """
        <div class="section">
            <h2> Cache Performance Analysis</h2>
            <div class="metric-grid">
"""
            html += f"""
                <div class="metric-card">
                    <div class="metric-value">{cache_efficiency.get('hit_rate', 0)*100:.1f}%</div>
                    <div class="metric-label">Hit Rate</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{cache_efficiency.get('miss_rate', 0)*100:.1f}%</div>
                    <div class="metric-label">Miss Rate</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{cache_efficiency.get('eviction_rate', 0)*100:.1f}%</div>
                    <div class="metric-label">Eviction Rate</div>
                </div>
"""
            html += """
            </div>
        </div>
"""
        
        # Recommendations section
        if recommendations:
            html += """
        <div class="section">
            <h2>💡 Optimization Recommendations</h2>
"""
            for rec in recommendations:
                priority = rec.get('priority', 'Medium').lower()
                html += f"""
            <div class="recommendation {priority}">
                <strong>[{rec.get('priority', 'Medium')}] {rec.get('category', 'General')}</strong><br>
                <strong>Issue:</strong> {rec.get('issue', '')}<br>
                <strong>Suggestion:</strong> {rec.get('suggestion', '')}
            </div>
"""
            html += """
        </div>
"""
        
        # Stage analysis section
        if stage_analysis and stage_analysis.get("stages"):
            html += """
        <div class="section">
            <h2>⏱️ Pipeline Stage Analysis</h2>
            <table class="stage-table">
                <thead>
                    <tr>
                        <th>Stage</th>
                        <th>Total Time (s)</th>
                        <th>Avg Time (s)</th>
                        <th>Call Count</th>
                        <th>% of Total</th>
                    </tr>
                </thead>
                <tbody>
"""
            for stage_name, stage_data in stage_analysis["stages"].items():
                html += f"""
                    <tr>
                        <td>{stage_name}</td>
                        <td>{stage_data.get('total_time', 0):.3f}</td>
                        <td>{stage_data.get('avg_time', 0):.3f}</td>
                        <td>{stage_data.get('call_count', 0)}</td>
                        <td>{stage_data.get('percentage_of_total', 0):.1f}%</td>
                    </tr>
"""
            html += """
                </tbody>
            </table>
        </div>
"""
        
        html += """
        <div class="section">
            <h2>ℹ️ Session Information</h2>
            <p><strong>Session ID:</strong> {session_id}</p>
            <p><strong>Start Time:</strong> {start_time}</p>
            <p><strong>End Time:</strong> {end_time}</p>
            <p><strong>Duration:</strong> {duration:.2f} seconds</p>
            <p><strong>Tags:</strong> {tags}</p>
        </div>
    </div>
</body>
</html>
""".format(
            session_id=session.get('id', 'Unknown'),
            start_time=datetime.fromtimestamp(session.get('start_time', 0)).strftime('%Y-%m-%d %H:%M:%S'),
            end_time=datetime.fromtimestamp(session.get('end_time', 0)).strftime('%Y-%m-%d %H:%M:%S') if session.get('end_time') else 'N/A',
            duration=session.get('duration', 0),
            tags=json.dumps(session.get('tags', {}))
        )
        
        return html
    
    def _save_raw_data(self, session_id: str, gpu_data: List[MetricSnapshot], 
                      cpu_data: List[MetricSnapshot]) -> Optional[str]:
        """Save raw monitoring data."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"raw_profiling_data_{session_id}_{timestamp}.json"
            filepath = os.path.join(self.output_dir, filename)
            
            raw_data = {
                "session_id": session_id,
                "gpu_data": [{"timestamp": s.timestamp, "metrics": s.metrics} for s in gpu_data],
                "cpu_data": [{"timestamp": s.timestamp, "metrics": s.metrics} for s in cpu_data],
                "saved_at": time.time()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(raw_data, f, indent=2, default=str)
            
            return filepath
        except Exception as e:
            print(f"Error saving raw data: {e}")
            return None