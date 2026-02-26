"""Interactive generation types for Inferix framework.

This module defines the core data types for interactive video generation,
supporting async queue-based interaction with consumer GPU constraints.

Key Concepts:
- InputApplyPolicy: When to apply user input (NEXT_SEGMENT recommended)
- QueuedInput: User input with estimated wait time
- InteractiveSession: Session manager for input queue and state
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List


class InputApplyPolicy(Enum):
    """Policy for when to apply user input during generation.
    
    NEXT_SEGMENT (Recommended):
        - Apply at segment boundary
        - Clean prompt transition, smooth overlap
        - Better for consumer GPUs with latency constraints
    
    NEXT_BLOCK:
        - Apply at block boundary
        - Faster response, may cause visual discontinuity
        - Use only when low latency is critical
    """
    NEXT_SEGMENT = "next_segment"
    NEXT_BLOCK = "next_block"


class InputState(Enum):
    """State of a queued user input."""
    QUEUED = "queued"           # Waiting for checkpoint
    PENDING = "pending"         # Will apply at next checkpoint
    APPLIED = "applied"         # Already applied
    DISCARDED = "discarded"     # Replaced by newer input


class SessionState(Enum):
    """State of an interactive generation session."""
    IDLE = "idle"               # Session not started
    GENERATING = "generating"   # Currently generating
    PAUSED = "paused"           # User paused generation
    COMPLETED = "completed"     # Generation finished
    ERROR = "error"             # Error occurred


class ControlCommand(Enum):
    """Control command for generation flow."""
    CONTINUE = "continue"       # Continue generation
    PAUSE = "pause"             # Pause generation (keep state)
    RESUME = "resume"           # Resume from pause
    STOP = "stop"               # Stop and discard
    MODIFY_PARAMS = "modify"    # Modify parameters (prompt, guidance, etc.)


@dataclass
class QueuedInput:
    """User input queued for async application.
    
    Attributes:
        input_id: Unique identifier for this input
        prompt: New text prompt (None = keep current)
        guidance_scale: New guidance scale (None = keep current)
        control: Control command for generation flow
        state: Current state of this input
        apply_policy: When to apply this input
        estimated_wait_seconds: Estimated time until application
        will_apply_at: Description of where input will be applied (e.g., "Segment 2")
        queued_at: Timestamp when input was queued (set by session)
    """
    input_id: str
    prompt: Optional[str] = None
    guidance_scale: Optional[float] = None
    control: ControlCommand = ControlCommand.CONTINUE
    state: InputState = InputState.QUEUED
    apply_policy: InputApplyPolicy = InputApplyPolicy.NEXT_SEGMENT
    estimated_wait_seconds: float = 0.0
    will_apply_at: str = ""
    queued_at: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "input_id": self.input_id,
            "prompt": self.prompt,
            "guidance_scale": self.guidance_scale,
            "control": self.control.value if self.control else None,
            "state": self.state.value,
            "apply_policy": self.apply_policy.value,
            "estimated_wait_seconds": self.estimated_wait_seconds,
            "will_apply_at": self.will_apply_at,
        }


@dataclass
class GenerationStatus:
    """Real-time generation status for UI display.
    
    Attributes:
        session_id: Unique session identifier
        state: Current session state
        current_segment: Current segment index (0-based)
        total_segments: Total segments to generate
        current_block: Current block index within segment (0-based)
        total_blocks: Total blocks per segment
        frames_generated: Total frames generated so far
        gpu_memory_gb: Current GPU memory usage in GB
        estimated_remaining_seconds: Estimated time to completion
        current_prompt: Currently active prompt (truncated)
        current_guidance: Currently active guidance scale
        queued_inputs: List of pending user inputs
        message: Human-readable status message
    """
    session_id: str
    state: SessionState
    current_segment: int = 0
    total_segments: int = 1
    current_block: int = 0
    total_blocks: int = 7
    frames_generated: int = 0
    gpu_memory_gb: float = 0.0
    estimated_remaining_seconds: float = 0.0
    current_prompt: str = ""
    current_guidance: float = 7.5
    queued_inputs: List[QueuedInput] = field(default_factory=list)
    message: str = ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "state": self.state.value,
            "progress": {
                "segment": f"{self.current_segment + 1}/{self.total_segments}",
                "block": f"{self.current_block + 1}/{self.total_blocks}",
                "frames": self.frames_generated,
            },
            "memory": {
                "gpu_gb": round(self.gpu_memory_gb, 2),
            },
            "timing": {
                "eta_seconds": round(self.estimated_remaining_seconds, 1),
            },
            "current_params": {
                "prompt": self.current_prompt[:100] + "..." if len(self.current_prompt) > 100 else self.current_prompt,
                "guidance": self.current_guidance,
            },
            "queued_inputs": [inp.to_dict() for inp in self.queued_inputs],
            "message": self.message,
        }
    
    @property
    def progress_percent(self) -> float:
        """Calculate overall progress percentage."""
        if self.total_segments == 0:
            return 0.0
        segment_progress = self.current_segment / self.total_segments
        block_progress = self.current_block / self.total_blocks / self.total_segments
        return (segment_progress + block_progress) * 100


@dataclass
class CheckpointResult:
    """Result from checkpoint evaluation.
    
    Returned by _evaluate_checkpoint() to indicate what action
    the pipeline should take at a segment/block boundary.
    
    Attributes:
        should_continue: Whether to continue generation
        pending_input: Input to apply (if any)
        new_prompt: New prompt to use (if changed)
        new_guidance: New guidance scale (if changed)
        command: Control command to execute
    """
    should_continue: bool = True
    pending_input: Optional[QueuedInput] = None
    new_prompt: Optional[str] = None
    new_guidance: Optional[float] = None
    command: ControlCommand = ControlCommand.CONTINUE


@dataclass
class SegmentBoundary:
    """Validated segment boundary information.
    
    Attributes:
        segment_idx: Segment index
        start_frame: Global start frame index
        end_frame: Global end frame index (inclusive)
        unique_frames: Number of unique frames in this segment
        overlap_with_previous: Frames overlapping with previous segment
        is_first: Whether this is the first segment
        is_last: Whether this is the last segment
    """
    segment_idx: int
    start_frame: int
    end_frame: int
    unique_frames: int
    overlap_with_previous: int
    is_first: bool
    is_last: bool
    
    def validate_initial_latent(self, initial_latent_frames: int, expected_overlap: int) -> bool:
        """Validate that initial latent matches expected overlap.
        
        Args:
            initial_latent_frames: Number of frames in initial_latent tensor
            expected_overlap: Expected overlap frames from config
            
        Returns:
            True if valid, False otherwise
        """
        if self.is_first:
            # First segment should have no initial latent
            return initial_latent_frames == 0
        else:
            # Non-first segments should have overlap frames
            return initial_latent_frames == expected_overlap


def calculate_total_frames(num_segments: int, segment_length: int, overlap_frames: int) -> int:
    """Calculate total unique frames for a multi-segment video.
    
    Formula: num_segments * segment_length - (num_segments - 1) * overlap_frames
    
    Args:
        num_segments: Number of segments
        segment_length: Frames per segment
        overlap_frames: Overlap between consecutive segments
        
    Returns:
        Total unique frames
    """
    if num_segments <= 0:
        return 0
    if num_segments == 1:
        return segment_length
    return num_segments * segment_length - (num_segments - 1) * overlap_frames


def validate_overlap_config(overlap_frames: int, block_size: int) -> bool:
    """Validate overlap configuration.
    
    Args:
        overlap_frames: Number of overlap frames
        block_size: Model's block size (e.g., 3 for Self-Forcing)
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If overlap is invalid
    """
    if overlap_frames < 0:
        raise ValueError(f"overlap_frames must be non-negative, got {overlap_frames}")
    
    if overlap_frames > 0 and overlap_frames % block_size != 0:
        raise ValueError(
            f"overlap_frames ({overlap_frames}) must be divisible by block_size ({block_size})"
        )
    
    return True
