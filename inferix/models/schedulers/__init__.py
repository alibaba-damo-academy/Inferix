"""Scheduler interfaces and base implementations for Inferix framework.

This module provides:
- SchedulerInterface: Abstract base class for all schedulers
- FlowMatchScheduler: Base implementation for flow matching

Model-specific schedulers should inherit from these base classes.
"""

from .flow_match import SchedulerInterface, FlowMatchScheduler

__all__ = ["SchedulerInterface", "FlowMatchScheduler"]