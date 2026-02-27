from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Callable, Tuple, Union, TYPE_CHECKING
from contextlib import nullcontext, contextmanager
import torch

# Profiling imports - profiling is a core module and should always be available
from ..profiling.profiler import InferixProfiler
from ..profiling.config import ProfilingConfig
from ..core.types import DecodeMode, MemoryMode, InputApplyPolicy, ControlCommand, SegmentBoundary, calculate_total_frames, validate_overlap_config, CheckpointResult, SessionState
from ..core.memory import AsyncMemoryManager, Granularity

if TYPE_CHECKING:
    from ..core.interactive import InteractiveSession


class AbstractInferencePipeline(ABC):
    """
    Abstract base class for all model inference pipelines.

    It defines a unified lifecycle and execution interface, ensuring the extensibility and consistency of the framework.
    All specific inference processes (such as Self-forcing, Causvid, Magi) should inherit from this class.
    """

    def __init__(self, config: Dict[str, Any], profiling_config: Optional[ProfilingConfig] = None):
        """
        Initialize the Pipeline.

        Args:
            config (Dict[str, Any]): Configuration dictionary loaded from YAML or JSON.
            profiling_config (Optional[ProfilingConfig]): Profiling configuration. If provided and enabled,
                                                          the pipeline will automatically collect performance metrics.
        """
        self.config = config
        self.model = None
        self._is_setup = False
        
        # Initialize profiling functionality
        self._profiling_config = profiling_config
        self._profiler: Optional[InferixProfiler] = None
        self._profiling_enabled = False
        
        # Check if profiling is enabled
        if profiling_config is not None and getattr(profiling_config, 'enabled', False):
            try:
                self._profiler = InferixProfiler(profiling_config)
                self._profiling_enabled = self._profiler.config.enabled if self._profiler else False
            except Exception as e:
                print(f"Warning: Failed to initialize profiler: {e}")
                self._profiler = None
                self._profiling_enabled = False
        else:
            self._profiler = None
            self._profiling_enabled = False
        
        # Memory manager for component-level offload (optional, experimental)
        self._memory_manager: Optional[AsyncMemoryManager] = None
        
        # Note: Subclass may print more specific initialization messages

    def _get_profiler_context(self, stage_name: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Get profiler context manager.
        
        Args:
            stage_name: Stage name
            metadata: Optional metadata
            
        Returns:
            Context manager (returns real profiler context if profiling is enabled, otherwise nullcontext)
        """
        if self._profiling_enabled and self._profiler is not None:
            return self._profiler.stage(stage_name, metadata)
        return nullcontext()

    def _start_profiling_session(self, session_id: str, tags: Optional[Dict[str, Any]] = None) -> str:
        """
        Start a new profiling session.
        
        Args:
            session_id: Session ID
            tags: Optional tag data
            
        Returns:
            Session ID
        """
        if self._profiling_enabled and self._profiler is not None:
            return self._profiler.start_session(session_id, tags)
        return session_id

    def _end_profiling_session(self) -> Optional[str]:
        """
        End current profiling session.
        
        Returns:
            Session ID (if there is an active session)
        """
        if self._profiling_enabled and self._profiler is not None:
            return self._profiler.end_session()
        return None

    def _add_profiling_event(self, event_name: str, data: Optional[Dict[str, Any]] = None):
        """
        Add a custom event to current session.
        
        Args:
            event_name: Event name
            data: Optional event data
        """
        if self._profiling_enabled and self._profiler is not None:
            self._profiler.add_event(event_name, data)

    def _initialize_pipeline(self):
        """
        [Template Method] Implementation of pipeline initialization.

        Each subclass can override this method to define its specific initialization logic.
        The default implementation is empty, and subclasses can add specific logic as needed.
        """
        pass

    @abstractmethod
    def load_checkpoint(self, checkpoint_path: str, **kwargs) -> None:
        """
        [Template Method] Load model checkpoint weights.

        Each subclass must implement this method to define how to load model weights.

        Args:
            checkpoint_path (str): Checkpoint file path.
            **kwargs: Other optional parameters.
        """
        pass
    
    def setup_devices(
        self, 
        low_memory: bool = False, 
        verbose: bool = False,
        use_memory_manager: bool = False
    ):
        """
        [Template Method] Setup devices with meta device optimization.
        
        Universal strategy for all pipelines:
        1. Materialize models from meta device to CPU
        2. Load checkpoint weights (if available)
        3. Move models to GPU layer-by-layer
        
        Subclasses should implement _get_model_components() to specify their models.
        
        Args:
            low_memory (bool): Enable CPU offloading for memory-constrained GPUs.
            verbose (bool): Enable detailed memory usage logs during loading.
            use_memory_manager (bool): Use AsyncMemoryManager for component-level offload.
                                       Requires >= 24GB VRAM. Will auto-fallback to
                                       DynamicSwapInstaller on smaller GPUs.
        """
        import gc
        import torch
        from inferix.core.memory.utils import gpu as get_gpu, get_cuda_free_memory_gb
        
        gpu = get_gpu()
        checkpoint = getattr(self, '_checkpoint_state_dict', None)
        
        # Check if memory manager is suitable for this GPU
        total_memory_gb = torch.cuda.get_device_properties(gpu).total_memory / (1024**3)
        if use_memory_manager and total_memory_gb < 24:
            print(f"⚠️  AsyncMemoryManager requires >= 24GB VRAM (detected: {total_memory_gb:.1f}GB)")
            print(f"   Falling back to DynamicSwapInstaller...")
            use_memory_manager = False
        
        # Show memory info only in verbose/low_memory mode
        log_memory = verbose or low_memory
        if log_memory:
            print(f"Initial GPU memory: {get_cuda_free_memory_gb(gpu):.2f} GB free")
        
        # Get model components from subclass
        model_components = self._get_model_components()
        
        # Track component info for memory manager registration
        component_layer_info = self._get_component_layer_info()
        
        for name, model in model_components.items():
            if log_memory:
                print(f"Loading {name}...")
            
            # Use memory manager mode or traditional mode
            if use_memory_manager and low_memory:
                self._materialize_for_memory_manager(model, name, gpu, checkpoint)
            else:
                self._materialize_and_load(model, name, gpu, checkpoint, low_memory)
            
            torch.cuda.empty_cache()
            gc.collect()
            if log_memory:
                print(f"After {name}: {get_cuda_free_memory_gb(gpu):.2f} GB free")
        
        # Initialize memory manager if requested and suitable
        if use_memory_manager and low_memory:
            self._init_memory_manager(gpu, model_components, component_layer_info)
        
        print("✅ All models loaded successfully")
    
    def _get_component_layer_info(self) -> Dict[str, Optional[List[str]]]:
        """
        [Template Method] Return layer attribute names for each component.
        
        Subclasses can override to specify which attributes contain layers
        for fine-grained memory management.
        
        Returns:
            Dict mapping component name to list of layer attribute names.
            Example: {'generator': ['model.blocks'], 'vae': None}
        """
        return {}
    
    def _materialize_for_memory_manager(self, model, name: str, gpu, checkpoint):
        """
        Materialize model for AsyncMemoryManager mode.
        
        In this mode, models are loaded to CPU first, then registered with
        the memory manager for explicit GPU/CPU swapping.
        """
        import torch
        
        model.to_empty(device='cpu')
        
        # Load checkpoint if this is the generator
        if checkpoint and name == 'generator':
            if hasattr(model, 'load_state_dict'):
                model.load_state_dict(checkpoint, assign=True)
        
        # Convert to bfloat16 AFTER loading checkpoint (checkpoint may have float32 params)
        model.to(dtype=torch.bfloat16)
        
        # Keep on CPU for now - memory manager will handle GPU placement
        # except for models that need to be on GPU immediately
        if name in ['text_encoder']:  # Text encoder is small, keep on GPU
            model.to(device=gpu)
    
    def _init_memory_manager(
        self, 
        gpu: torch.device,
        components: Dict[str, Any],
        layer_info: Dict[str, Optional[List[str]]]
    ):
        """
        Initialize AsyncMemoryManager with registered components.
        """
        from inferix.core.memory.utils import get_cuda_free_memory_gb
        
        # Create memory manager with auto-detected budget
        self._memory_manager = AsyncMemoryManager(
            device=gpu,
            budget_gb=get_cuda_free_memory_gb(gpu) * 0.85
        )
        
        # Register each component
        for name, model in components.items():
            layer_names = layer_info.get(name)
            self._memory_manager.register_component(name, model, layer_names)
        
        # Load generator to GPU as initial state (needed for diffusion first)
        if 'generator' in components:
            self._memory_manager.load_async('generator')
            self._memory_manager.wait('generator')
            print(f"  [MemoryManager] Generator loaded to GPU as initial state")
        
        print(f"✅ AsyncMemoryManager initialized (budget: {self._memory_manager.budget_gb:.1f} GB)")
    
    @contextmanager
    def _use_component(self, *names: str):
        """
        Context manager to ensure specified components are on GPU.
        
        Use this in inference code to automatically manage component placement.
        Only active when memory manager is enabled.
        
        Args:
            *names: Component names to load to GPU
            
        Example:
            with self._use_component('generator'):
                latent = generator.denoise(...)
        """
        if self._memory_manager is not None:
            with self._memory_manager.use(*names):
                yield
        else:
            yield
    
    @contextmanager
    def _exclusive_component(self, name: str):
        """
        Context manager for exclusive component access (offloads others).
        
        Use this when you need maximum memory for a specific component,
        e.g., VAE decode.
        
        Args:
            name: Component name to keep on GPU exclusively
            
        Example:
            with self._exclusive_component('vae'):
                video = vae.decode(latent)
        """
        if self._memory_manager is not None:
            with self._memory_manager.exclusive(name):
                yield
        else:
            yield
    
    def _prefetch_component(self, name: str):
        """
        Start loading a component in background.
        
        Use this to overlap data transfer with computation.
        
        Args:
            name: Component name to prefetch
        """
        if self._memory_manager is not None:
            self._memory_manager.prefetch(name)
    
    def _get_model_components(self) -> Dict[str, Any]:
        """
        [Template Method] Return model components to be loaded.
        
        Subclasses must override this to specify their models.
        
        Returns:
            Dict mapping component name to model object.
            Example: {'text_encoder': self.pipeline.text_encoder, 'generator': ...}
        """
        return {}
    
    def _materialize_and_load(self, model, name: str, gpu, checkpoint, low_memory: bool):
        """
        [Concrete Method] Materialize model from meta device and load to GPU.
        
        This is the universal loading logic. Subclasses can override for custom behavior.
        """
        import torch
        from inferix.core.memory.utils import DynamicSwapInstaller
        
        model.to_empty(device='cpu')
        
        # Load checkpoint weights first (before dtype conversion to avoid mismatch)
        if checkpoint is not None:
            model.load_state_dict(checkpoint, assign=True)
        
        # Convert to bfloat16 after loading weights
        model.to(dtype=torch.bfloat16)
        
        if low_memory:
            DynamicSwapInstaller.install_model(model, device=gpu)
        else:
            model.to(device=gpu)

    def setup(self):
        """
        [Concrete Method] Execute a unified setup process.

        This method is generic for all subclasses. It is responsible for calling the _initialize_pipeline method
        to execute specific initialization logic.
        To prevent duplicate initialization, the internal state `_is_setup` will be checked.
        """
        # Wrap entire setup process with profiling context
        with self._get_profiler_context("pipeline_setup"):
            if self._is_setup:
                return

            self._initialize_pipeline()
            self._is_setup = True

    @abstractmethod
    def run_text_to_video(self, prompts: List[str], **kwargs) -> Any:
        """
        [Template Method] Execute text-to-video inference logic.

        Each subclass must implement this method to define its specific text-to-video generation logic.

        Args:
            prompts (List[str]): Input text prompts.
            **kwargs: Other parameters.

        Returns:
            Any: Generated video result.
        """
        pass

    @abstractmethod
    def run_image_to_video(self, prompts: List[str], image_path: str, **kwargs) -> Any:
        """
        [Template Method] Execute image-to-video inference logic.

        Each subclass must implement this method to define its specific image-to-video generation logic.

        Args:
            prompts (List[str]): Input text prompts.
            image_path (str): Input image path.
            **kwargs: Other parameters.

        Returns:
            Any: Generated video result.
        """
        pass

    def run(self, inputs: Dict[str, Any], **kwargs) -> Any:
        """
        [Template Method] Execute core inference logic and return results.

        Call the corresponding inference method based on input type.

        Args:
            inputs (Dict[str, Any]): A dictionary containing all necessary inputs.
            **kwargs: Other optional runtime parameters.

        Returns:
            Any: Final result of inference.
        """
        # Wrap entire run process with profiling context
        with self._get_profiler_context("pipeline_run"):
            # Default implementation can call corresponding inference method based on input content
            if 'prompts' in inputs and 'image_path' in inputs:
                return self.run_image_to_video(inputs['prompts'], inputs['image_path'], **kwargs)
            elif 'prompts' in inputs:
                return self.run_text_to_video(inputs['prompts'], **kwargs)
            elif 'prompt' in inputs and 'image_path' in inputs:
                # Backward compatibility: support single prompt
                return self.run_image_to_video([inputs['prompt']], inputs['image_path'], **kwargs)
            elif 'prompt' in inputs:
                # Backward compatibility: support single prompt
                return self.run_text_to_video([inputs['prompt']], **kwargs)
            else:
                raise ValueError("Invalid inputs for pipeline execution")

    def __call__(self, **kwargs) -> Any:
        """
        [Concrete Method] Provide a unified and convenient entry point.

        It first ensures that `setup` has been called, then executes the `run` method.
        This way, users can directly use the pipeline instance like calling a function.
        
        For example: `pipeline(prompt="a dog running")`

        Args:
            **kwargs: Parameters passed to the `run` method.

        Returns:
            Any: Return result of the `run` method.
        """
        # Wrap entire __call__ process with profiling context
        with self._get_profiler_context("pipeline_call"):
            if not self._is_setup:
                self.setup()
            
            # Pass kwargs as inputs dict to run method
            return self.run(inputs=kwargs)

    def cleanup_profiling(self):
        """
        Clean up profiling resources.
        Subclasses can call this method to clean up profiling resources when appropriate.
        """
        if self._profiling_enabled and self._profiler is not None:
            self._profiler.cleanup()
    
    def run_streaming_generation(
        self,
        prompts: List[str],
        stream_callback: Optional[Callable[[torch.Tensor], None]] = None,
        num_segments: int = 1,
        segment_length: int = 21,
        overlap_frames: int = 3,
        **kwargs
    ) -> Optional[torch.Tensor]:
        """
        [Framework-level Method] Streaming video generation with universal interface.
        
        TERMINOLOGY:
        ------------
        - BLOCK: Model-specific generation unit (e.g., 3 frames per block in Self-Forcing).
                 Blocks are generated sequentially using KV cache for autoregressive generation.
                 This is an INTERNAL detail of each model's architecture.
        
        - SEGMENT: Framework-level generation unit (e.g., 21 frames = 7 blocks in Self-Forcing).
                   A segment is a complete generation cycle that can be streamed progressively.
                   Multiple segments can be chained together for long videos.
        
        STREAMING HIERARCHY:
        -------------------
        1. BLOCK-LEVEL (Inner Loop - Model Implementation):
           - Self-Forcing generates 3 frames per block using semi-autoregressive decoding
           - Each block is decoded and streamed immediately after generation
           - Callback triggered: block_callback(block_latent, block_index)
        
        2. SEGMENT-LEVEL (Outer Loop - Framework Management):
           - A segment contains multiple blocks (e.g., 21 frames = 7 blocks)
           - After each segment, memory is cleaned up for long-video generation
           - Segments can overlap for smooth transitions
        
        USAGE MODES:
        -----------
        Mode 1: Single-Segment Block-Wise Streaming
            Use Case: Short video (e.g., 21 frames) with real-time streaming
            Example:
                from inferix.core.media import create_streaming_backend
                
                streamer = create_streaming_backend("gradio")
                streamer.connect(width=832, height=480, fps=16)
                
                pipeline.run_streaming_generation(
                    prompts=['a cat walking'],
                    stream_callback=streamer.stream_batch,
                    num_segments=1,
                    segment_length=21  # 7 blocks × 3 frames/block
                )
            Flow:
                Block 0 (frames 0-2)   → decode → stream → continue
                Block 1 (frames 3-5)   → decode → stream → continue
                ...
                Block 6 (frames 18-20) → decode → stream → done
        
        Mode 2: Multi-Segment Long-Video Streaming
            Use Case: Long video (e.g., 210 frames) with memory management
            Example:
                pipeline.run_streaming_generation(
                    prompts=['a cat walking'],
                    stream_callback=streamer.stream_batch,
                    num_segments=10,        # 10 segments
                    segment_length=21,      # 21 frames per segment
                    overlap_frames=3        # 3 frames overlap between segments
                )
            Flow:
                Segment 0: blocks 0-6   → 21 frames (0-20)   → cleanup → continue
                Segment 1: blocks 0-6   → 21 frames (18-38)  → cleanup → continue
                                          ↑ overlap 3 frames
                ...
                Segment 9: blocks 0-6   → 21 frames          → cleanup → done
            Total: 10×21 - 9×3 = 183 unique frames
        
        Args:
            prompts (List[str]): Text prompts for generation.
            stream_callback (Optional[Callable]): Callback function for streaming decoded frames.
                                                  Signature: callback(frames: torch.Tensor)
                                                  - frames: shape [T, H, W, C], dtype uint8, range [0, 255]
            num_segments (int): Number of segments to generate.
                               - 1 = short video with block-wise streaming only
                               - >1 = long video with segment looping and memory cleanup
            segment_length (int): Number of frames per segment.
                                 - Must match model's block size requirements
                                 - Self-Forcing: 21 frames (7 blocks × 3 frames/block)
            overlap_frames (int): Number of overlapping frames between consecutive segments.
                                 - Used for smooth transitions in long videos
                                 - Must be divisible by block size (e.g., 3 for Self-Forcing)
                                 - Only applies when num_segments > 1
            **kwargs: Additional model-specific parameters:
                     - num_samples: batch size
                     - low_memory: enable memory optimization
        
        Returns:
            Optional[torch.Tensor]: Generated video tensor [B, T, H, W, C].
                                   Can be None if only streaming is needed (video already sent).
        
        Raises:
            ValueError: If segment_length doesn't match model's block size requirements.
            NotImplementedError: If the specific pipeline hasn't implemented streaming yet.
        
        Note:
            - Block-wise streaming is implemented by each model's _generate_segment_with_streaming()
            - Segment-level looping is handled by this framework method
            - For long video testing, use num_segments=10-20
        """
        # Print generation configuration summary
        self._print_generation_config(
            num_prompts=len(prompts),
            num_segments=num_segments,
            segment_length=segment_length,
            overlap_frames=overlap_frames,
            **kwargs
        )
        
        print(f"[Streaming] Starting generation: {num_segments} segment(s) × {segment_length} frames")
        
        all_videos = []
        initial_latent = None
        
        for segment_idx in range(num_segments):
            # Use cyclic prompts if there are fewer prompts than segments
            current_prompt = prompts[segment_idx % len(prompts)]
            
            print(f"[Streaming] Segment {segment_idx + 1}/{num_segments}...")
            
            # Call subclass implementation for single segment generation
            video_segment, final_latent = self._generate_segment_with_streaming(
                prompt=current_prompt,
                initial_latent=initial_latent,
                stream_callback=stream_callback,
                segment_length=segment_length,
                **kwargs
            )
            
            all_videos.append(video_segment)
            
            # Extract overlapping frames for next segment
            if segment_idx < num_segments - 1:
                initial_latent = final_latent[:, -overlap_frames:]
            
            # Clean up memory between segments
            self._cleanup_segment_memory()
        
        print(f"✅ Streaming generation completed: {num_segments} segment(s)")
        
        # Return concatenated video if multiple segments
        return torch.cat(all_videos, dim=1) if len(all_videos) > 1 else all_videos[0] if all_videos else None
    
    @abstractmethod
    def _generate_segment_with_streaming(
        self,
        prompt: str,
        initial_latent: Optional[torch.Tensor],
        stream_callback: Optional[Callable[[torch.Tensor], None]],
        segment_length: int = 21,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        [Abstract Method] Generate a single segment with progressive block-wise streaming.
        
        IMPLEMENTATION RESPONSIBILITY:
        ------------------------------
        Each model pipeline must implement this method according to its architecture:
        
        - Self-Forcing: Semi-autoregressive generation with block-wise decoding
          * Block size: 3 frames (num_frame_per_block=3)
          * Segment length: 21 frames = 7 blocks
          * Implementation: Generate each block → decode → stream → continue
        
        - CausVid: Rollout-based generation with chunk streaming
          * Block size: varies based on configuration
          * Implementation: Generate each rollout → decode → stream → continue
        
        - Magi: Chunk-based generation
          * Block size: varies based on configuration
          * Implementation: Generate each chunk → decode → stream → continue
        
        BLOCK vs SEGMENT (from model's perspective):
        --------------------------------------------
        - BLOCK: Your model's atomic generation unit
          * Defined by model architecture (e.g., num_frame_per_block)
          * Generated using KV cache for autoregressive continuation
          * Example (Self-Forcing): 3 frames per block
        
        - SEGMENT: One complete generation cycle requested by the framework
          * Defined by user/framework (segment_length parameter)
          * Contains multiple blocks: segment_length / block_size blocks
          * Example (Self-Forcing): 21 frames = 7 blocks
        
        STREAMING WORKFLOW:
        ------------------
        1. Initialize generation state (KV cache, etc.)
        2. For each block in the segment:
           a. Generate block latent (e.g., 3 frames)
           b. Decode block to pixel space immediately
           c. Call stream_callback(decoded_frames) if provided
           d. Update KV cache for next block
        3. Return (full_segment_video, final_latent)
        
        EXAMPLE IMPLEMENTATION (Self-Forcing):
        --------------------------------------
        def _generate_segment_with_streaming(self, prompt, initial_latent, 
                                            stream_callback, segment_length, **kwargs):
            # Validate segment_length is multiple of block_size
            assert segment_length % self.num_frame_per_block == 0
            
            # Define block callback for progressive decoding
            def block_callback(block_latent, block_index):
                # Decode this block immediately
                block_video = self.vae.decode(block_latent)
                # Stream via provided callback
                if stream_callback:
                    stream_callback(block_video)
            
            # Generate with block callback
            latents = self.pipeline.inference(
                ...,
                block_callback=block_callback  # Hook into block generation
            )
            
            # Return full segment and final latent for continuation
            return full_video, final_latent
        
        Args:
            prompt (str): Text prompt for current segment generation.
            initial_latent (Optional[torch.Tensor]): Initial latent state for continuation.
                                                     - None: start new video (T2V mode)
                                                     - [B, overlap_frames, C, H, W]: continue from previous segment
            stream_callback (Optional[Callable]): Callback for streaming decoded blocks.
                                                 - Called after each block is decoded
                                                 - Signature: callback(frames: torch.Tensor)
                                                 - frames: [T, H, W, C], uint8, range [0, 255]
            segment_length (int): Total number of frames to generate in this segment.
                                 - Must be compatible with model's block size
                                 - Will be validated by implementation
            **kwargs: Model-specific parameters:
                     - num_samples: batch size for generation
                     - low_memory: enable memory optimization mode
                     - guidance_scale: classifier-free guidance scale
                     - etc.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (segment_video, final_latent)
                - segment_video: Decoded video frames for this segment
                  * Shape: [B, segment_length, H, W, C]
                  * dtype: torch.float32
                  * Range: [0, 1]
                
                - final_latent: Final latent state after this segment
                  * Shape: [B, segment_length, C, H, W] (full latent space)
                  * Used as initial_latent for next segment
                  * Framework will extract overlap_frames automatically
        
        Raises:
            ValueError: If segment_length is incompatible with model's block size.
            NotImplementedError: If streaming is not yet implemented for this model.
        
        Note:
            - This method is called by run_streaming_generation() for each segment
            - Block-level streaming happens inside this method
            - Memory cleanup happens between segments (handled by framework)
        """
        pass
    
    def _cleanup_segment_memory(self):
        """
        Clean up memory between segments.
        
        Default implementation clears CUDA cache. Subclasses can override
        for model-specific cleanup (e.g., clearing KV cache).
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # =========================================================================
    # Interactive Generation Methods
    # =========================================================================
    
    def run_interactive_generation(
        self,
        session: "InteractiveSession",
        initial_prompt: str,
        num_segments: int = 1,
        segment_length: int = 21,
        overlap_frames: int = 3,
        stream_callback: Optional[Callable[[torch.Tensor], None]] = None,
        block_size: int = 3,
        **kwargs
    ) -> Optional[torch.Tensor]:
        """
        [Framework-level Method] Interactive video generation with async queue-based input.
        
        This method provides fundamental interactive generation capability,
        replacing run_streaming_generation's cyclic prompts with InteractiveSession.
        
        KEY FEATURES:
        - Async queue-based input (not real-time, suitable for consumer GPUs)
        - Checkpoint evaluation at segment boundaries
        - Pause/Resume/Stop control
        - Status broadcast to UI
        - Comprehensive boundary validation
        
        BOUNDARY HANDLING:
        - Overlap frames extracted correctly at segment transitions
        - Prompt changes applied at segment boundaries only (NEXT_SEGMENT policy)
        - KV cache should be cleared by subclass when prompt changes
        
        USAGE:
            session = InteractiveSession(apply_policy=InputApplyPolicy.NEXT_SEGMENT)
            session.set_initial_prompt("A cat walking")
            session.set_status_callback(ui_backend.broadcast_status)
            
            video = pipeline.run_interactive_generation(
                session=session,
                initial_prompt="A cat walking",
                num_segments=10,
                segment_length=21,
                overlap_frames=3,
                stream_callback=streamer.stream_batch,
            )
        
        Args:
            session: InteractiveSession managing user input and state
            initial_prompt: Initial text prompt for generation
            num_segments: Number of segments to generate
            segment_length: Frames per segment (must be multiple of block_size)
            overlap_frames: Overlapping frames between segments (must be multiple of block_size)
            stream_callback: Optional callback for streaming decoded frames
            block_size: Model's block size for boundary validation
            **kwargs: Additional model-specific parameters
            
        Returns:
            Generated video tensor [B, T, H, W, C], or None if stopped early
            
        Raises:
            ValueError: If boundary configuration is invalid
        """
        # Import here to avoid circular import
        from ..core.interactive import InteractiveSession
        
        # Validate boundary configuration
        self._validate_boundary_config(
            segment_length=segment_length,
            overlap_frames=overlap_frames,
            block_size=block_size,
            num_segments=num_segments,
        )
        
        # Initialize session
        session.set_initial_prompt(initial_prompt)
        session.set_generation_params(
            total_segments=num_segments,
            blocks_per_segment=segment_length // block_size,
        )
        session.start_session()
        
        # Print configuration
        self._print_generation_config(
            num_prompts=1,  # Session manages prompts dynamically
            num_segments=num_segments,
            segment_length=segment_length,
            overlap_frames=overlap_frames,
            interactive=True,
            **kwargs
        )
        
        print(f"[Interactive] Session {session.session_id}: {num_segments} segments, {calculate_total_frames(num_segments, segment_length, overlap_frames)} frames")
        
        all_videos = []
        initial_latent = None
        current_prompt = initial_prompt
        current_guidance = kwargs.get('guidance_scale', 7.5)
        
        try:
            for segment_idx in range(num_segments):
                # Validate segment boundary
                boundary = self._validate_segment_boundary(
                    segment_idx=segment_idx,
                    num_segments=num_segments,
                    overlap_frames=overlap_frames,
                    segment_length=segment_length,
                    block_size=block_size,
                    initial_latent=initial_latent,
                )
                
                # Evaluate checkpoint at segment start
                checkpoint_result = session.evaluate_checkpoint(
                    checkpoint_type="segment",
                    checkpoint_index=segment_idx,
                    current_prompt=current_prompt,
                    current_guidance=current_guidance,
                )
                
                # Handle control commands
                if checkpoint_result.command == ControlCommand.STOP:
                    print(f"[Interactive] Stop requested at segment {segment_idx}")
                    break
                
                # Apply parameter changes
                if checkpoint_result.new_prompt:
                    print(f"[Interactive] Prompt changed: '{current_prompt[:30]}...' -> '{checkpoint_result.new_prompt[:30]}...'")
                    current_prompt = checkpoint_result.new_prompt
                
                if checkpoint_result.new_guidance:
                    print(f"[Interactive] Guidance changed: {current_guidance} -> {checkpoint_result.new_guidance}")
                    current_guidance = checkpoint_result.new_guidance
                
                # Wait if paused
                while session.should_pause():
                    if session.should_stop():
                        break
                    session.wait_for_resume(timeout=0.1)
                
                if session.should_stop():
                    break
                
                print(f"[Interactive] Segment {segment_idx + 1}/{num_segments}...")
                
                # Generate segment
                video_segment, final_latent = self._generate_segment_with_streaming(
                    prompt=current_prompt,
                    initial_latent=initial_latent,
                    stream_callback=stream_callback,
                    segment_length=segment_length,
                    guidance_scale=current_guidance,
                    **kwargs
                )
                
                all_videos.append(video_segment)
                
                # Update session progress
                frames_generated = sum(v.shape[1] for v in all_videos)
                gpu_memory_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                session.update_progress(
                    segment_idx=segment_idx,
                    block_idx=segment_length // block_size - 1,
                    frames_generated=frames_generated,
                    gpu_memory_gb=gpu_memory_gb,
                )
                
                # Extract overlap for next segment
                if segment_idx < num_segments - 1:
                    initial_latent = self._extract_overlap_latent(
                        final_latent=final_latent,
                        overlap_frames=overlap_frames,
                        segment_idx=segment_idx,
                    )
                
                # Clean up memory
                self._cleanup_segment_memory()
        
        except Exception as e:
            session.end_session(error=True)
            raise
        
        finally:
            session.end_session(error=False)
        
        # Concatenate results
        if len(all_videos) == 0:
            return None
        
        result = torch.cat(all_videos, dim=1) if len(all_videos) > 1 else all_videos[0]
        print(f"[Interactive] Generation completed: {result.shape[1]} frames")
        
        return result
    
    def _validate_boundary_config(
        self,
        segment_length: int,
        overlap_frames: int,
        block_size: int,
        num_segments: int,
    ):
        """Validate boundary configuration before generation.
        
        Args:
            segment_length: Frames per segment
            overlap_frames: Overlapping frames between segments
            block_size: Model's block size
            num_segments: Number of segments
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate segment_length
        if segment_length <= 0:
            raise ValueError(f"segment_length must be positive, got {segment_length}")
        
        if segment_length % block_size != 0:
            raise ValueError(
                f"segment_length ({segment_length}) must be divisible by block_size ({block_size})"
            )
        
        # Validate overlap_frames
        validate_overlap_config(overlap_frames, block_size)
        
        if overlap_frames >= segment_length:
            raise ValueError(
                f"overlap_frames ({overlap_frames}) must be less than segment_length ({segment_length})"
            )
        
        # Validate num_segments
        if num_segments <= 0:
            raise ValueError(f"num_segments must be positive, got {num_segments}")
        
        # Warn about potential issues
        if num_segments > 1 and overlap_frames == 0:
            print("[Warning] No overlap between segments. Transitions may be discontinuous.")
    
    def _validate_segment_boundary(
        self,
        segment_idx: int,
        num_segments: int,
        overlap_frames: int,
        segment_length: int,
        block_size: int,
        initial_latent: Optional[torch.Tensor],
    ) -> SegmentBoundary:
        """Validate and compute segment boundary information.
        
        Args:
            segment_idx: Current segment index (0-based)
            num_segments: Total number of segments
            overlap_frames: Overlapping frames between segments
            segment_length: Frames per segment
            block_size: Model's block size
            initial_latent: Initial latent for this segment (None for first segment)
            
        Returns:
            SegmentBoundary with validated information
            
        Raises:
            ValueError: If boundary is invalid
        """
        is_first = (segment_idx == 0)
        is_last = (segment_idx == num_segments - 1)
        
        # Calculate frame indices
        if is_first:
            start_frame = 0
            end_frame = segment_length - 1
            unique_frames = segment_length
            overlap_with_previous = 0
        else:
            # Non-first segments start after overlap
            start_frame = segment_idx * segment_length - segment_idx * overlap_frames
            end_frame = start_frame + segment_length - 1
            unique_frames = segment_length - overlap_frames
            overlap_with_previous = overlap_frames
        
        # Validate initial_latent
        if initial_latent is not None:
            if is_first:
                raise ValueError(
                    f"First segment (idx=0) should have initial_latent=None, "
                    f"got tensor with shape {initial_latent.shape}"
                )
            
            latent_frames = initial_latent.shape[1]
            if latent_frames != overlap_frames:
                raise ValueError(
                    f"initial_latent has {latent_frames} frames, expected {overlap_frames} "
                    f"overlap frames for segment {segment_idx}"
                )
        elif not is_first:
            raise ValueError(
                f"Non-first segment (idx={segment_idx}) requires initial_latent with "
                f"{overlap_frames} frames, got None"
            )
        
        return SegmentBoundary(
            segment_idx=segment_idx,
            start_frame=start_frame,
            end_frame=end_frame,
            unique_frames=unique_frames,
            overlap_with_previous=overlap_with_previous,
            is_first=is_first,
            is_last=is_last,
        )
    
    def _extract_overlap_latent(
        self,
        final_latent: torch.Tensor,
        overlap_frames: int,
        segment_idx: int,
    ) -> torch.Tensor:
        """Extract overlap frames from final latent for next segment.
        
        Args:
            final_latent: Final latent tensor [B, T, C, H, W]
            overlap_frames: Number of overlap frames to extract
            segment_idx: Current segment index (for logging)
            
        Returns:
            Overlap latent tensor [B, overlap_frames, C, H, W]
            
        Raises:
            ValueError: If extraction fails
        """
        if overlap_frames <= 0:
            raise ValueError(f"overlap_frames must be positive, got {overlap_frames}")
        
        if final_latent is None:
            raise ValueError("final_latent is None, cannot extract overlap")
        
        total_frames = final_latent.shape[1]
        if total_frames < overlap_frames:
            raise ValueError(
                f"final_latent has {total_frames} frames, cannot extract {overlap_frames} overlap frames"
            )
        
        # Extract last overlap_frames
        overlap_latent = final_latent[:, -overlap_frames:]
        
        # Validate shape
        assert overlap_latent.shape[1] == overlap_frames, \
            f"Extracted {overlap_latent.shape[1]} frames, expected {overlap_frames}"
        
        print(f"[Interactive] Extracted {overlap_frames} overlap frames for segment {segment_idx + 2}")
        
        return overlap_latent
    
    def _print_generation_config(
        self,
        num_prompts: int,
        num_segments: int,
        segment_length: int,
        overlap_frames: int = 0,
        **kwargs
    ):
        """
        Print generation configuration summary before video generation.
        
        This method is called by run_streaming_generation() and run_interactive_generation()
        and can be overridden by subclasses to add model-specific configuration details.
        
        Args:
            num_prompts: Number of prompts
            num_segments: Number of segments to generate
            segment_length: Frames per segment
            overlap_frames: Overlapping frames between segments
            **kwargs: Additional model-specific parameters
                - interactive: If True, indicate interactive mode
        """
        # Extract common parameters
        num_samples = kwargs.get('num_samples', 1)
        low_memory = kwargs.get('low_memory', False)
        decode_mode = kwargs.get('decode_mode', 'AFTER_ALL')
        memory_mode = kwargs.get('memory_mode', 'balanced')
        chunk_size = kwargs.get('chunk_size', None)
        interactive = kwargs.get('interactive', False)
        
        # Normalize mode names
        if hasattr(decode_mode, 'value'):
            decode_mode = decode_mode.value
        if hasattr(memory_mode, 'value'):
            memory_mode = memory_mode.value
        
        # Calculate effective chunk_size based on memory_mode if not explicitly set
        if chunk_size is None:
            memory_mode_lower = memory_mode.lower() if isinstance(memory_mode, str) else 'balanced'
            if memory_mode_lower == 'aggressive':
                effective_chunk_size = 2
            elif memory_mode_lower == 'relaxed':
                effective_chunk_size = 7
            else:  # balanced
                effective_chunk_size = 4
        else:
            effective_chunk_size = chunk_size
        
        # Calculate total frames
        if num_segments > 1:
            total_frames = num_segments * segment_length - (num_segments - 1) * overlap_frames
        else:
            total_frames = segment_length
        
        print("\n" + "=" * 60)
        mode_str = "Interactive Generation" if interactive else "Generation Configuration"
        print(mode_str)
        print("=" * 60)
        if interactive:
            print(f"Mode:            Interactive (async queue)")
        print(f"Prompts:         {num_prompts}")
        print(f"Batch Size:      {num_samples}")
        print(f"Total Frames:    {total_frames} ({num_segments} seg x {segment_length} frames)")
        if num_segments > 1:
            print(f"Overlap:         {overlap_frames} frames")
        
        # Allow subclasses to add model-specific info
        # Pass segment_length and num_segments to subclass
        self._print_model_specific_config(
            segment_length=segment_length,
            num_segments=num_segments,
            **kwargs
        )
        
        # Framework-level modes
        print("-" * 60)
        print(f"Decode Mode:     {decode_mode}")
        print(f"Memory Mode:     {memory_mode}")
        chunk_source = "(preset)" if chunk_size is None else "(custom)"
        print(f"VAE Chunk Size:  {effective_chunk_size} frames {chunk_source}")
        if low_memory:
            print(f"Low Memory:      Enabled")
        print("=" * 60 + "\n")
    
    def _print_model_specific_config(self, **kwargs):
        """
        Print model-specific configuration details.
        
        Subclasses can override this to add custom configuration info.
        Default implementation does nothing.
        
        Args:
            **kwargs: Model-specific parameters
        """
        pass
    
    def _apply_memory_mode(self, mode: MemoryMode, vae_chunk_size: Optional[int] = None):
        """
        Apply memory management strategy.
        
        MemoryMode provides preset combinations for common scenarios.
        vae_chunk_size can be explicitly set to override the preset.
        
        Priority: explicit vae_chunk_size > MemoryMode preset > default
        
        Args:
            mode: Memory management mode (AGGRESSIVE, BALANCED, RELAXED)
            vae_chunk_size: Optional explicit chunk size override
        """
        # Apply MemoryMode presets
        if mode == MemoryMode.AGGRESSIVE:
            # Free cache before VAE, smaller chunks (16GB VRAM)
            self._free_cache_before_vae = True
            preset_chunk_size = 2
        elif mode == MemoryMode.RELAXED:
            # Keep cache for reuse, larger chunks (24GB+ VRAM)
            self._free_cache_before_vae = False
            preset_chunk_size = 7
        else:  # BALANCED
            self._free_cache_before_vae = True
            preset_chunk_size = 4
        
        # Allow explicit override
        self._vae_chunk_size = vae_chunk_size if vae_chunk_size is not None else preset_chunk_size
    
    def _decode_latent(
        self,
        latent: torch.Tensor,
        vae,
        decode_mode: DecodeMode = DecodeMode.AFTER_ALL,
        chunk_size: int = 2,
        stream_callback: Optional[Callable[[torch.Tensor], None]] = None,
        block_size: int = 3,
    ) -> Optional[torch.Tensor]:
        """
        Framework-level VAE decoding with strategy support.
        
        NOTE: block_size vs chunk_size
        - block_size: Model's generation unit (e.g., 3 frames for Self-Forcing)
                     Used to split latent for progressive decoding
        - chunk_size: VAE's internal decode batch size (e.g., 2 frames)
                     Used within VAE to reduce peak memory
        
        Args:
            latent: Latent tensor [B, T, C, H, W]
            vae: VAE model with decode_to_pixel method
            decode_mode: Decoding strategy
            chunk_size: Frames per VAE internal decode batch (memory optimization)
            stream_callback: Callback for streaming decoded frames
            block_size: Model's block size for PER_BLOCK mode (generation unit)
        
        Returns:
            Decoded video tensor, or None if NO_DECODE
        """
        if decode_mode == DecodeMode.NO_DECODE:
            return None
        
        if decode_mode == DecodeMode.AFTER_ALL:
            # Decode all at once with chunked caching for memory efficiency
            video = vae.decode_to_pixel(latent, use_cache=True, chunk_size=chunk_size)
            video = (video * 0.5 + 0.5).clamp(0, 1)
            return video
        
        if decode_mode == DecodeMode.PER_BLOCK:
            # Decode per block (split by block_size, each block decoded with chunk_size)
            num_frames = latent.shape[1]
            videos = []
            for start in range(0, num_frames, block_size):
                end = min(start + block_size, num_frames)
                block_latent = latent[:, start:end]
                # Each block decoded independently with chunked caching
                block_video = vae.decode_to_pixel(block_latent, use_cache=True, chunk_size=chunk_size)
                block_video = (block_video * 0.5 + 0.5).clamp(0, 1)
                if stream_callback:
                    stream_callback(block_video)
                videos.append(block_video)
                torch.cuda.empty_cache()
            return torch.cat(videos, dim=1)
        
        return None