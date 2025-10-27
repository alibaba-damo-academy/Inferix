"""Diffusion model specific analysis for Inferix profiler."""

from typing import Dict, Any, List, Optional


class DiffusionAnalyzer:
    """Analyzer for diffusion model specific metrics."""
    
    def __init__(self, base_profiler):
        """Initialize the diffusion analyzer.
        
        Args:
            base_profiler: The base InferixProfiler instance
        """
        self.base_profiler = base_profiler
        self.diffusion_steps = []
        self.model_parameters = {}
        self.block_computations = []
        
    def record_diffusion_step(self, step: int, timestep: float, 
                            block_size: int, computation_time_ms: float,
                            guidance_scale: Optional[float] = None):
        """Record metrics for a diffusion step.
        
        Args:
            step: Diffusion step number
            timestep: Timestep value (0.0 to 1.0)
            block_size: Number of frames per block
            computation_time_ms: Time taken for this step in milliseconds
            guidance_scale: Classifier guidance scale used
        """
        if not self.base_profiler.config.enabled or not self.base_profiler.config.profile_diffusion_steps:
            return
            
        step_data = {
            'step': step,
            'timestep': timestep,
            'block_size': block_size,
            'computation_time_ms': computation_time_ms,
            'guidance_scale': guidance_scale
        }
        
        self.diffusion_steps.append(step_data)
        
        # Add as a custom event
        self.base_profiler.add_event("diffusion_step", step_data)
    
    def record_model_parameters(self, model_name: str, parameters_count: int, 
                              model_type: str):
        """Record model parameter count.
        
        Args:
            model_name: Name of the model
            parameters_count: Number of parameters
            model_type: Type of model (diffusion, text_encoder, vae, etc.)
        """
        if not self.base_profiler.config.enabled or not self.base_profiler.config.profile_model_parameters:
            return
            
        self.model_parameters[model_name] = {
            'parameters_count': parameters_count,
            'model_type': model_type
        }
        
        # Add as a custom event
        self.base_profiler.add_event("model_parameters", {
            'model_name': model_name,
            'parameters_count': parameters_count,
            'model_type': model_type
        })
    
    def record_block_computation(self, block_index: int, block_size: int,
                               computation_time_ms: float, memory_usage_mb: float):
        """Record metrics for a block computation.
        
        Args:
            block_index: Index of the block
            block_size: Number of frames in the block
            computation_time_ms: Time taken for this block in milliseconds
            memory_usage_mb: Peak memory usage during block computation in MB
        """
        if not self.base_profiler.config.enabled or not self.base_profiler.config.profile_block_computation:
            return
            
        block_data = {
            'block_index': block_index,
            'block_size': block_size,
            'computation_time_ms': computation_time_ms,
            'memory_usage_mb': memory_usage_mb
        }
        
        self.block_computations.append(block_data)
        
        # Add as a custom event
        self.base_profiler.add_event("block_computation", block_data)
    
    def get_diffusion_analysis(self) -> Optional[Dict[str, Any]]:
        """Get analysis of diffusion steps.
        
        Returns:
            Dictionary with diffusion analysis metrics or None if no data
        """
        if not self.diffusion_steps:
            return None
            
        total_steps = len(self.diffusion_steps)
        total_time = sum(step['computation_time_ms'] for step in self.diffusion_steps)
        avg_time = total_time / total_steps if total_steps > 0 else 0
        
        # Calculate average timestep and block size
        avg_timestep = sum(step['timestep'] for step in self.diffusion_steps) / total_steps
        avg_block_size = sum(step['block_size'] for step in self.diffusion_steps) / total_steps
        
        # Find min/max values
        min_time = min(step['computation_time_ms'] for step in self.diffusion_steps)
        max_time = max(step['computation_time_ms'] for step in self.diffusion_steps)
        
        # Calculate FPS and BPS
        total_frames = sum(step['block_size'] for step in self.diffusion_steps)
        total_time_seconds = total_time / 1000.0  # Convert ms to seconds
        
        fps = total_frames / total_time_seconds if total_time_seconds > 0 else 0
        bps = total_steps / total_time_seconds if total_time_seconds > 0 else 0
        
        return {
            'total_steps': total_steps,
            'total_computation_time_ms': total_time,
            'avg_computation_time_ms': avg_time,
            'min_computation_time_ms': min_time,
            'max_computation_time_ms': max_time,
            'avg_timestep': avg_timestep,
            'avg_block_size': avg_block_size,
            'total_frames': total_frames,
            'fps': fps,
            'bps': bps,
            'steps': self.diffusion_steps
        }
    
    def get_model_analysis(self) -> Optional[Dict[str, Any]]:
        """Get analysis of model parameters.
        
        Returns:
            Dictionary with model analysis metrics or None if no data
        """
        if not self.model_parameters:
            return None
            
        total_parameters = sum(model['parameters_count'] for model in self.model_parameters.values())
        
        # Format readable parameter count
        if total_parameters >= 1e9:
            total_parameters_readable = f"{total_parameters / 1e9:.1f}B"
        elif total_parameters >= 1e6:
            total_parameters_readable = f"{total_parameters / 1e6:.1f}M"
        else:
            total_parameters_readable = f"{total_parameters:,}"
            
        return {
            'models': self.model_parameters,
            'total_parameters': total_parameters,
            'total_parameters_readable': total_parameters_readable
        }
    
    def get_block_analysis(self) -> Optional[Dict[str, Any]]:
        """Get analysis of block computations.
        
        Returns:
            Dictionary with block analysis metrics or None if no data
        """
        if not self.block_computations:
            return None
            
        total_blocks = len(self.block_computations)
        total_time = sum(block['computation_time_ms'] for block in self.block_computations)
        avg_time = total_time / total_blocks if total_blocks > 0 else 0
        
        # Calculate average block size
        avg_block_size = sum(block['block_size'] for block in self.block_computations) / total_blocks
        total_frames = sum(block['block_size'] for block in self.block_computations)
        
        # Memory analysis
        avg_memory = sum(block['memory_usage_mb'] for block in self.block_computations) / total_blocks
        peak_memory = max(block['memory_usage_mb'] for block in self.block_computations)
        
        # Calculate FPS and BPS for blocks
        total_time_seconds = total_time / 1000.0  # Convert ms to seconds
        fps = total_frames / total_time_seconds if total_time_seconds > 0 else 0
        bps = total_blocks / total_time_seconds if total_time_seconds > 0 else 0
        
        return {
            'total_blocks': total_blocks,
            'total_computation_time_ms': total_time,
            'avg_computation_time_ms': avg_time,
            'avg_block_size': avg_block_size,
            'total_frames': total_frames,
            'avg_memory_usage_mb': avg_memory,
            'peak_memory_usage_mb': peak_memory,
            'fps': fps,
            'bps': bps,
            'blocks': self.block_computations
        }
    
    def get_performance_recommendations(self) -> List[Dict[str, str]]:
        """Get performance optimization recommendations based on analysis.
        
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        # Diffusion steps recommendations
        diffusion_analysis = self.get_diffusion_analysis()
        if diffusion_analysis and diffusion_analysis['total_steps'] > 30:
            recommendations.append({
                'category': 'Diffusion Steps',
                'priority': 'High',
                'issue': f'Using {diffusion_analysis["total_steps"]} diffusion steps',
                'suggestion': f'Consider reducing steps to 20-30 to improve generation speed. Current average timestep progression: {diffusion_analysis["avg_timestep"]:.3f}.'
            })
        
        # Block size recommendations
        block_analysis = self.get_block_analysis()
        if block_analysis and block_analysis['avg_block_size'] > 8:
            recommendations.append({
                'category': 'Block Size',
                'priority': 'Medium',
                'issue': f'Average block size is {block_analysis["avg_block_size"]:.1f} frames',
                'suggestion': 'Consider reducing block size to improve latency and memory usage. Smaller blocks enable better streaming performance.'
            })
        
        # Computation time recommendations
        if diffusion_analysis and diffusion_analysis['avg_computation_time_ms'] > 200:
            recommendations.append({
                'category': 'Computation Time',
                'priority': 'High',
                'issue': f'Average step time is {diffusion_analysis["avg_computation_time_ms"]:.1f}ms',
                'suggestion': 'Optimize model architecture or reduce computational complexity to improve generation speed.'
            })
        
        # FPS recommendations
        if diffusion_analysis and diffusion_analysis['fps'] < 10:
            recommendations.append({
                'category': 'Generation Speed',
                'priority': 'High',
                'issue': f'Current FPS is {diffusion_analysis["fps"]:.1f}',
                'suggestion': f'Current generation speed is below real-time requirements. Consider optimizing the model or reducing the computational load to achieve at least 16 FPS for smooth video generation.'
            })
        
        # BPS recommendations
        if diffusion_analysis and diffusion_analysis['bps'] < 2:
            recommendations.append({
                'category': 'Block Processing',
                'priority': 'Medium',
                'issue': f'Current BPS is {diffusion_analysis["bps"]:.1f}',
                'suggestion': f'Current block processing speed is relatively slow. Consider optimizing block processing or using parallel processing to improve throughput.'
            })
        
        return recommendations
