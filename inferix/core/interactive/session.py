"""Interactive session management for async queue-based video generation.

This module provides the InteractiveSession class which manages:
- User input queue (FIFO, keep only latest pending input)
- Checkpoint hooks for pipeline integration
- Pause/resume/stop control
- Distributed synchronization (rank 0 broadcasts inputs)

Key Design Decisions:
- NEXT_SEGMENT policy is recommended for consumer GPUs
- Only one pending input is kept (newer inputs replace older ones)
- Estimated wait time is calculated based on current progress
"""

import threading
import time
import uuid
from typing import Optional, List, Callable, TYPE_CHECKING
from dataclasses import dataclass

import torch
import torch.distributed as dist

from inferix.core.types.interactive import (
    InputApplyPolicy,
    InputState,
    SessionState,
    ControlCommand,
    QueuedInput,
    GenerationStatus,
    CheckpointResult,
)

if TYPE_CHECKING:
    from inferix.models.wan_base.utils.parallel_config import ParallelConfig


class InteractiveSession:
    """Session manager for interactive video generation.
    
    This class manages the lifecycle of an interactive generation session,
    handling user input queuing, checkpoint evaluation, and state synchronization.
    
    Usage:
        session = InteractiveSession(apply_policy=InputApplyPolicy.NEXT_SEGMENT)
        session.set_initial_prompt("A cat walking")
        
        # In UI thread:
        session.submit_input(prompt="A dog running")
        
        # In generation thread:
        for segment_idx in range(num_segments):
            result = session.evaluate_checkpoint("segment", segment_idx, current_prompt)
            if result.command == ControlCommand.STOP:
                break
            if result.new_prompt:
                current_prompt = result.new_prompt
            # ... generate segment ...
            session.update_progress(segment_idx, block_idx, frames_generated)
    
    Thread Safety:
        - All public methods are thread-safe
        - Uses threading.Lock for state mutations
        - Uses threading.Event for pause/resume
    
    Distributed Support:
        - Rank 0 is the input source
        - Inputs are broadcast to all workers at checkpoints
        - Generation state is synchronized across workers
    """
    
    def __init__(
        self,
        apply_policy: InputApplyPolicy = InputApplyPolicy.NEXT_SEGMENT,
        parallel_config: Optional["ParallelConfig"] = None,
    ):
        """Initialize interactive session.
        
        Args:
            apply_policy: When to apply user input (default: NEXT_SEGMENT)
            parallel_config: Optional parallel config for distributed mode
        """
        self.session_id: str = str(uuid.uuid4())[:8]
        self._apply_policy = apply_policy
        self._parallel_config = parallel_config
        
        # Input queue (keep only latest pending)
        self._input_queue: List[QueuedInput] = []
        self._lock = threading.Lock()
        
        # Control state
        self._pause_event = threading.Event()
        self._pause_event.set()  # Initially not paused
        self._stop_flag = False
        
        # Generation state
        self._state = SessionState.IDLE
        self._current_segment = 0
        self._current_block = 0
        self._total_segments = 1
        self._blocks_per_segment = 7
        
        # Current parameters
        self._current_prompt: str = ""
        self._current_guidance: float = 7.5
        self._initial_prompt: str = ""
        
        # Status callback
        self._status_callback: Optional[Callable[[GenerationStatus], None]] = None
        
        # Timing
        self._session_start_time: float = 0.0
        self._segment_start_time: float = 0.0
        self._avg_time_per_block: float = 0.5  # Initial estimate (will be updated)
        
        # Device for distributed communication
        self._device: Optional[torch.device] = None
    
    @classmethod
    def from_prompts(
        cls,
        prompts: List[str],
        apply_policy: InputApplyPolicy = InputApplyPolicy.NEXT_SEGMENT,
        parallel_config: Optional["ParallelConfig"] = None,
    ) -> "InteractiveSession":
        """Create session from a list of prompts (backward compatibility).
        
        This allows using InteractiveSession with pre-defined prompts,
        similar to run_streaming_generation's cyclic prompts.
        
        Args:
            prompts: List of prompts (will be queued for each segment)
            apply_policy: When to apply input
            parallel_config: Optional parallel config
            
        Returns:
            InteractiveSession with prompts pre-queued
        """
        session = cls(apply_policy=apply_policy, parallel_config=parallel_config)
        if prompts:
            session.set_initial_prompt(prompts[0])
            # Queue remaining prompts for subsequent segments
            for i, prompt in enumerate(prompts[1:], start=1):
                session.submit_input(prompt=prompt)
        return session
    
    def set_initial_prompt(self, prompt: str, guidance: float = 7.5):
        """Set initial prompt for generation.
        
        Args:
            prompt: Initial text prompt
            guidance: Initial guidance scale
        """
        with self._lock:
            self._initial_prompt = prompt
            self._current_prompt = prompt
            self._current_guidance = guidance
    
    def set_status_callback(self, callback: Callable[[GenerationStatus], None]):
        """Set callback for status updates.
        
        Args:
            callback: Function to call with GenerationStatus
        """
        self._status_callback = callback
    
    def set_device(self, device: torch.device):
        """Set device for distributed communication.
        
        Args:
            device: Torch device (e.g., cuda:0)
        """
        self._device = device
    
    def set_generation_params(self, total_segments: int, blocks_per_segment: int):
        """Set generation parameters for progress tracking.
        
        Args:
            total_segments: Total number of segments to generate
            blocks_per_segment: Number of blocks per segment
        """
        with self._lock:
            self._total_segments = total_segments
            self._blocks_per_segment = blocks_per_segment
    
    # =========================================================================
    # User Input Submission
    # =========================================================================
    
    def submit_input(
        self,
        prompt: Optional[str] = None,
        guidance_scale: Optional[float] = None,
        control: ControlCommand = ControlCommand.CONTINUE,
    ) -> QueuedInput:
        """Submit user input for async application.
        
        This method is thread-safe and can be called from UI thread
        while generation is running in another thread.
        
        Args:
            prompt: New prompt (None = keep current)
            guidance_scale: New guidance scale (None = keep current)
            control: Control command (CONTINUE, PAUSE, STOP, etc.)
            
        Returns:
            QueuedInput with estimated wait time
        """
        with self._lock:
            # Discard older queued inputs (keep only latest)
            for inp in self._input_queue:
                if inp.state == InputState.QUEUED:
                    inp.state = InputState.DISCARDED
            
            # Create new input
            queued = QueuedInput(
                input_id=str(uuid.uuid4())[:8],
                prompt=prompt,
                guidance_scale=guidance_scale,
                control=control,
                state=InputState.QUEUED,
                apply_policy=self._apply_policy,
                queued_at=time.time(),
            )
            
            # Calculate estimated wait time
            queued.estimated_wait_seconds = self._estimate_wait_time()
            queued.will_apply_at = self._get_apply_target()
            
            self._input_queue.append(queued)
            
            # Handle control commands
            if control == ControlCommand.PAUSE:
                self._pause_event.clear()
            elif control == ControlCommand.RESUME:
                self._pause_event.set()
            elif control == ControlCommand.STOP:
                self._stop_flag = True
            
            return queued
    
    def _estimate_wait_time(self) -> float:
        """Estimate time until next checkpoint."""
        if self._apply_policy == InputApplyPolicy.NEXT_SEGMENT:
            # Time remaining in current segment
            remaining_blocks = self._blocks_per_segment - self._current_block
            return remaining_blocks * self._avg_time_per_block
        else:  # NEXT_BLOCK
            return self._avg_time_per_block
    
    def _get_apply_target(self) -> str:
        """Get description of where input will be applied."""
        if self._apply_policy == InputApplyPolicy.NEXT_SEGMENT:
            return f"Segment {self._current_segment + 1}"
        else:
            return f"Block {self._current_block + 1}"
    
    # =========================================================================
    # Checkpoint Evaluation (Called by Pipeline)
    # =========================================================================
    
    def evaluate_checkpoint(
        self,
        checkpoint_type: str,
        checkpoint_index: int,
        current_prompt: str,
        current_guidance: float = 7.5,
    ) -> CheckpointResult:
        """Evaluate checkpoint for user input and control commands.
        
        This method should be called by the pipeline at segment/block boundaries.
        
        Args:
            checkpoint_type: "segment" or "block"
            checkpoint_index: Current segment/block index
            current_prompt: Current active prompt
            current_guidance: Current guidance scale
            
        Returns:
            CheckpointResult with action to take
        """
        with self._lock:
            # Update generation state
            if checkpoint_type == "segment":
                self._current_segment = checkpoint_index
                self._current_block = 0
            else:
                self._current_block = checkpoint_index
            
            # Check for stop
            if self._stop_flag:
                return CheckpointResult(
                    should_continue=False,
                    command=ControlCommand.STOP,
                )
        
        # Wait if paused (outside lock to allow resume)
        while not self._pause_event.is_set():
            if self._stop_flag:
                return CheckpointResult(
                    should_continue=False,
                    command=ControlCommand.STOP,
                )
            self._pause_event.wait(timeout=0.1)
        
        with self._lock:
            # Check for applicable input
            should_apply = False
            if checkpoint_type == "segment":
                should_apply = (self._apply_policy == InputApplyPolicy.NEXT_SEGMENT)
            else:
                should_apply = (self._apply_policy == InputApplyPolicy.NEXT_BLOCK)
            
            if not should_apply:
                return CheckpointResult(should_continue=True)
            
            # Find latest queued input
            pending_input = None
            for inp in reversed(self._input_queue):
                if inp.state == InputState.QUEUED:
                    pending_input = inp
                    break
            
            if pending_input is None:
                return CheckpointResult(should_continue=True)
            
            # Mark as applied
            pending_input.state = InputState.APPLIED
            
            # Build result
            result = CheckpointResult(
                should_continue=True,
                pending_input=pending_input,
                command=pending_input.control,
            )
            
            if pending_input.prompt:
                result.new_prompt = pending_input.prompt
                self._current_prompt = pending_input.prompt
            
            if pending_input.guidance_scale:
                result.new_guidance = pending_input.guidance_scale
                self._current_guidance = pending_input.guidance_scale
            
            # Broadcast to workers in distributed mode
            self._broadcast_input_to_workers(pending_input)
            
            return result
    
    # =========================================================================
    # Progress Updates
    # =========================================================================
    
    def update_progress(
        self,
        segment_idx: int,
        block_idx: int,
        frames_generated: int,
        gpu_memory_gb: float = 0.0,
    ):
        """Update generation progress.
        
        Args:
            segment_idx: Current segment index
            block_idx: Current block index
            frames_generated: Total frames generated so far
            gpu_memory_gb: Current GPU memory usage
        """
        with self._lock:
            self._current_segment = segment_idx
            self._current_block = block_idx
            self._state = SessionState.GENERATING
        
        # Broadcast status
        self._broadcast_status(
            frames_generated=frames_generated,
            gpu_memory_gb=gpu_memory_gb,
        )
    
    def start_session(self):
        """Mark session as started."""
        with self._lock:
            self._state = SessionState.GENERATING
            self._session_start_time = time.time()
            self._segment_start_time = time.time()
    
    def end_session(self, error: bool = False):
        """Mark session as ended."""
        with self._lock:
            self._state = SessionState.ERROR if error else SessionState.COMPLETED
    
    def _broadcast_status(
        self,
        frames_generated: int,
        gpu_memory_gb: float,
    ):
        """Broadcast current status to callback."""
        if self._status_callback is None:
            return
        
        # Get queued inputs
        with self._lock:
            queued = [inp for inp in self._input_queue if inp.state == InputState.QUEUED]
        
        # Estimate remaining time
        remaining_segments = self._total_segments - self._current_segment - 1
        remaining_blocks = self._blocks_per_segment - self._current_block - 1
        total_remaining_blocks = remaining_segments * self._blocks_per_segment + remaining_blocks
        estimated_remaining = total_remaining_blocks * self._avg_time_per_block
        
        status = GenerationStatus(
            session_id=self.session_id,
            state=self._state,
            current_segment=self._current_segment,
            total_segments=self._total_segments,
            current_block=self._current_block,
            total_blocks=self._blocks_per_segment,
            frames_generated=frames_generated,
            gpu_memory_gb=gpu_memory_gb,
            estimated_remaining_seconds=estimated_remaining,
            current_prompt=self._current_prompt,
            current_guidance=self._current_guidance,
            queued_inputs=queued,
            message=f"Generating Segment {self._current_segment + 1}/{self._total_segments}",
        )
        
        self._status_callback(status)
    
    # =========================================================================
    # Control Methods
    # =========================================================================
    
    def should_pause(self) -> bool:
        """Check if generation should pause."""
        return not self._pause_event.is_set()
    
    def wait_for_resume(self, timeout: Optional[float] = None):
        """Wait for resume signal."""
        self._pause_event.wait(timeout)
    
    def should_stop(self) -> bool:
        """Check if generation should stop."""
        return self._stop_flag
    
    def pause(self):
        """Pause generation."""
        self._pause_event.clear()
    
    def resume(self):
        """Resume generation."""
        self._pause_event.set()
    
    def stop(self):
        """Stop generation."""
        self._stop_flag = True
    
    # =========================================================================
    # Distributed Support
    # =========================================================================
    
    def _broadcast_input_to_workers(self, input: QueuedInput):
        """Broadcast user input from rank 0 to all workers.
        
        This ensures all workers apply the same input at checkpoints.
        """
        if not dist.is_initialized():
            return
        
        if self._device is None:
            return
        
        # Serialize input
        if self._is_rank_0():
            input_data = f"{input.prompt or ''}|{input.guidance_scale or 0}|{input.control.value}"
            input_tensor = torch.tensor(
                [ord(c) for c in input_data],
                dtype=torch.uint8,
                device=self._device,
            )
            input_len = torch.tensor(len(input_tensor), device=self._device)
        else:
            input_len = torch.tensor(0, device=self._device)
        
        # Broadcast length
        dist.broadcast(input_len, src=0)
        
        # Broadcast content
        if not self._is_rank_0():
            input_tensor = torch.zeros(
                input_len.item(),
                dtype=torch.uint8,
                device=self._device,
            )
        dist.broadcast(input_tensor, src=0)
        
        # Deserialize on workers
        if not self._is_rank_0():
            input_data = ''.join(chr(c) for c in input_tensor.tolist())
            parts = input_data.split('|')
            if len(parts) >= 3:
                input.prompt = parts[0] if parts[0] else None
                input.guidance_scale = float(parts[1]) if parts[1] != '0' else None
                input.control = ControlCommand(parts[2])
    
    def _sync_generation_state(self):
        """Synchronize generation state across workers."""
        if not dist.is_initialized() or self._device is None:
            return
        
        # Broadcast state from rank 0
        state_tensor = torch.tensor(
            [self._current_segment, self._current_block],
            device=self._device,
        )
        dist.broadcast(state_tensor, src=0)
        
        # Update on workers
        if not self._is_rank_0():
            self._current_segment = state_tensor[0].item()
            self._current_block = state_tensor[1].item()
    
    def _is_rank_0(self) -> bool:
        """Check if current process is rank 0."""
        if self._parallel_config is None:
            return True
        return self._parallel_config.rank == 0
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def state(self) -> SessionState:
        """Current session state."""
        return self._state
    
    @property
    def current_prompt(self) -> str:
        """Current active prompt."""
        return self._current_prompt
    
    @property
    def current_guidance(self) -> float:
        """Current guidance scale."""
        return self._current_guidance
    
    @property
    def initial_prompt(self) -> str:
        """Initial prompt set for session."""
        return self._initial_prompt
    
    @property
    def apply_policy(self) -> InputApplyPolicy:
        """Input application policy."""
        return self._apply_policy
    
    @property
    def is_distributed(self) -> bool:
        """Whether running in distributed mode."""
        return dist.is_initialized()
