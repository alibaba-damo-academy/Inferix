import queue
from dataclasses import dataclass
from typing import Optional

import torch

from inferix.distributed import parallel_state as mpu


@dataclass
class TensorAndHandler:
    tensor: torch.Tensor
    handler: torch.distributed.Work


class PPScheduler:
    def __init__(self):
        """Initialize an instance of the PPScheduler class"""

        self.device: torch.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self.recv_queue: queue.Queue = queue.Queue()

    def isend_next(self, tensor: torch.Tensor) -> torch.distributed.Work:
        """Asynchronously send a tensor to the next pipeline and return the send handle.

        Args:
            tensor (torch.Tensor): The tensor to be sent.

        Returns:
            torch.distributed.Work: The handle for the send operation.
        """
        handle = torch.distributed.isend(
            tensor.contiguous(), dst=mpu.get_pipeline_model_parallel_next_rank(), group=mpu.get_pp_group()
        )
        return handle

    def irecv_prev(self, buffer: torch.Tensor) -> torch.distributed.Work:
        """Asynchronously receive a tensor from the previous pipeline and return the receive handle.

        Args:
            buffer (torch.Tensor): The buffer tensor for receiving data.

        Returns:
            torch.distributed.Work: The handle for the receive operation.
        """
        handle = torch.distributed.irecv(buffer, src=mpu.get_pipeline_model_parallel_prev_rank(), group=mpu.get_pp_group())
        return handle

    def recv_prev_data(self, shape: torch.Size, dtype: torch.dtype) -> torch.Tensor:
        """Receive data from the previous pipeline and return the received tensor.

        Args:
            shape (torch.Size): The shape of the tensor to receive.
            dtype (torch.dtype): The data type of the tensor to receive.

        Returns:
            torch.Tensor: The received tensor.
        """
        recv_tensor = torch.empty(shape, dtype=dtype, device=self.device)
        self.irecv_prev(recv_tensor).wait()
        return recv_tensor

    def queue_irecv_prev(self, shape: torch.Size, dtype: torch.dtype) -> None:
        """Put the asynchronously received tensor and handle into the receive queue.

        Args:
            shape (torch.Size): The shape of the tensor to receive.
            dtype (torch.dtype): The data type of the tensor to receive.
        """
        recv_tensor = torch.empty(shape, dtype=dtype, device=self.device)
        handle = self.irecv_prev(recv_tensor)
        self.recv_queue.put(TensorAndHandler(tensor=recv_tensor, handler=handle))

    def queue_irecv_prev_data(self) -> torch.Tensor:
        """Get a tensor from the receive queue and wait for the receive operation to complete.

        Returns:
            torch.Tensor: The received tensor obtained from the queue.
        """
        tensor_and_handler = self.recv_queue.get()
        tensor_and_handler.handler.wait()
        return tensor_and_handler.tensor


_PP_SCHEDULER: Optional[PPScheduler] = None


def init_pp_scheduler():
    """Initialize the PPScheduler instance.

    Raises:
        AssertionError: If the PPScheduler is already initialized.
    """
    global _PP_SCHEDULER
    assert _PP_SCHEDULER is None, "pipeline model parallel group is already initialized"
    _PP_SCHEDULER = PPScheduler()


def pp_scheduler() -> PPScheduler:
    """Get the current PPScheduler instance.

    Returns:
        PPScheduler: The current PPScheduler instance.

    Raises:
        AssertionError: If the PPScheduler has not been initialized.
    """
    assert _PP_SCHEDULER is not None, "pipeline model parallel group is not initialized"
    return _PP_SCHEDULER
