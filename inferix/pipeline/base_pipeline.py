from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from contextlib import nullcontext

# Profiling imports - profiling is a core module and should always be available
from ..profiling.profiler import InferixProfiler
from ..profiling.config import ProfilingConfig


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
        
        # 初始化profiling功能
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
        
        print(f"Initializing {self.__class__.__name__}{' with profiling enabled' if self._profiling_enabled else ''}...")

    def _get_profiler_context(self, stage_name: str, metadata: Optional[Dict[str, Any]] = None):
        """
        获取profiler上下文管理器。
        
        Args:
            stage_name: 阶段名称
            metadata: 可选的元数据
            
        Returns:
            上下文管理器（如果profiling启用则返回真实的profiler上下文，否则返回nullcontext）
        """
        if self._profiling_enabled and self._profiler is not None:
            return self._profiler.stage(stage_name, metadata)
        return nullcontext()

    def _start_profiling_session(self, session_id: str, tags: Optional[Dict[str, Any]] = None) -> str:
        """
        开始一个新的profiling会话。
        
        Args:
            session_id: 会话ID
            tags: 可选的标签数据
            
        Returns:
            会话ID
        """
        if self._profiling_enabled and self._profiler is not None:
            return self._profiler.start_session(session_id, tags)
        return session_id

    def _end_profiling_session(self) -> Optional[str]:
        """
        结束当前的profiling会话。
        
        Returns:
            会话ID（如果有活跃会话）
        """
        if self._profiling_enabled and self._profiler is not None:
            return self._profiler.end_session()
        return None

    def _add_profiling_event(self, event_name: str, data: Optional[Dict[str, Any]] = None):
        """
        添加一个自定义事件到当前会话。
        
        Args:
            event_name: 事件名称
            data: 可选的事件数据
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

    def setup(self):
        """
        [Concrete Method] Execute a unified setup process.

        This method is generic for all subclasses. It is responsible for calling the _initialize_pipeline method
        to execute specific initialization logic.
        To prevent duplicate initialization, the internal state `_is_setup` will be checked.
        """
        # 使用profiling上下文包装整个setup过程
        with self._get_profiler_context("pipeline_setup"):
            if self._is_setup:
                print(f"{self.__class__.__name__} is already set up.")
                return

            print(f"Setting up {self.__class__.__name__}...")
            self._initialize_pipeline()
            
            self._is_setup = True
            print("Setup complete.")

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
        # 使用profiling上下文包装整个run过程
        with self._get_profiler_context("pipeline_run"):
            # 默认实现可以根据输入内容调用相应的推理方法
            if 'prompts' in inputs and 'image_path' in inputs:
                return self.run_image_to_video(inputs['prompts'], inputs['image_path'], **kwargs)
            elif 'prompts' in inputs:
                return self.run_text_to_video(inputs['prompts'], **kwargs)
            elif 'prompt' in inputs and 'image_path' in inputs:
                # 向后兼容：支持单个prompt
                return self.run_image_to_video([inputs['prompt']], inputs['image_path'], **kwargs)
            elif 'prompt' in inputs:
                # 向后兼容：支持单个prompt
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
        # 使用profiling上下文包装整个__call__过程
        with self._get_profiler_context("pipeline_call"):
            if not self._is_setup:
                print("Setup has not been called. Calling it now...")
                self.setup()
            
            # 将kwargs作为inputs字典传递给run方法
            return self.run(inputs=kwargs)

    def cleanup_profiling(self):
        """
        清理profiling资源。
        子类可以在适当的时候调用此方法来清理profiling资源。
        """
        if self._profiling_enabled and self._profiler is not None:
            self._profiler.cleanup()