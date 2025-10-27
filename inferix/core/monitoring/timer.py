from datetime import datetime

import torch


class EventPathTimer:
    """
    A lightweight class for recording time without any distributed barrier.

    This class allows for recording elapsed time between events without requiring
    synchronization across distributed processes. It maintains the previous message
    and time to calculate the duration between consecutive records.
    """

    def __init__(self):
        """
        Initialize the EventPathTimer.

        This constructor sets the previous message and time to None, preparing
        the instance for recording events.
        """
        self.prev_message: str = None
        self.prev_time: datetime = None

    def reset(self):
        """
        Reset the recorded message and time.

        This method clears the previous message and time, allowing for a fresh
        start in recording new events.
        """
        self.prev_message = None
        self.prev_time = None

    def synced_record(self, message):
        """
        Record the current time with a message.

        Args:
            message (str): A message to log along with the current time.

        This method synchronizes the CUDA operations, records the current time,
        and calculates the elapsed time since the last recorded message, if any.
        It then logs the elapsed time along with the previous and current messages.
        """
        torch.cuda.synchronize()
        current_time = datetime.now()
        if self.prev_message is not None:
            # Import here to avoid circular import
            from ...distributed.dist_utils import print_rank_0
            print_rank_0(
                f"\nTime Elapsed: [{current_time - self.prev_time}] From [{self.prev_message} ({self.prev_time})] To [{message} ({current_time})]"
            )
        self.prev_message = message
        self.prev_time = current_time


_GLOBAL_LIGHT_TIMER = EventPathTimer()


def event_path_timer() -> EventPathTimer:
    """Get the current EventPathTimer instance.

    Returns:
        EventPathTimer: The current EventPathTimer instance.

    Raises:
        AssertionError: If the EventPathTimer has not been initialized.
    """
    assert _GLOBAL_LIGHT_TIMER is not None, "light time recorder is not initialized"
    return _GLOBAL_LIGHT_TIMER