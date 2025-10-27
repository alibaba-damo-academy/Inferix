"""Test script for Inferix profiling module.

This script tests the core functionality of the profiling system,
including GPU/CPU monitoring, session management, and report generation.
"""

import sys
import os
import time
import tempfile
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from inferix.profiling import InferixProfiler, ProfilingConfig
from inferix.profiling.monitors import GPUMonitor, CPUMonitor
from inferix.core.monitoring.profiling import ProfilingIntegration


def test_profiling_config():
    """Test profiling configuration."""
    print("Testing ProfilingConfig...")
    
    # Test default config
    config = ProfilingConfig()
    assert not config.enabled
    assert config.gpu_monitor_interval == 0.5
    assert config.cpu_monitor_interval == 1.0
    
    # Test custom config
    config = ProfilingConfig(
        enabled=True,
        gpu_monitor_interval=0.2,
        output_dir="/tmp/test",
        session_tags={"test": "value"}
    )
    assert config.enabled
    assert config.gpu_monitor_interval == 0.2
    assert config.output_dir == "/tmp/test"
    assert config.session_tags["test"] == "value"
    
    print("✓ ProfilingConfig tests passed")


def test_monitors():
    """Test GPU and CPU monitors."""
    print("Testing monitors...")
    
    # Test CPU monitor (should always work)
    try:
        cpu_monitor = CPUMonitor(interval=0.1, max_data_points=10)
        cpu_monitor.start_monitoring()
        time.sleep(0.5)  # Let it collect some data
        cpu_monitor.stop_monitoring()
        
        data = cpu_monitor.get_all_data()
        assert len(data) > 0, "CPU monitor should collect data"
        
        latest = cpu_monitor.get_latest_metrics()
        assert latest is not None, "Should have latest metrics"
        assert 'cpu_usage_percent' in latest.metrics
        assert 'memory_used_mb' in latest.metrics
        
        print("✓ CPU monitor test passed")
    except Exception as e:
        print(f"✗ CPU monitor test failed: {e}")
        return False
    
    # Test GPU monitor (may not work on all systems)
    try:
        gpu_monitor = GPUMonitor(interval=0.1, max_data_points=10)
        gpu_monitor.start_monitoring()
        time.sleep(0.5)  # Let it collect some data
        gpu_monitor.stop_monitoring()
        
        data = gpu_monitor.get_all_data()
        if len(data) > 0:
            print("✓ GPU monitor test passed")
        else:
            print("⚠ GPU monitor test: No GPU data collected (GPU may not be available)")
    except Exception as e:
        print(f"⚠ GPU monitor test: {e} (GPU monitoring may not be available)")
    
    return True


def test_profiler():
    """Test the main profiler functionality."""
    print("Testing InferixProfiler...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = ProfilingConfig(
            enabled=True,
            gpu_monitor_interval=0.1,
            cpu_monitor_interval=0.1,
            real_time_display=False,  # Disable for testing
            generate_final_report=True,
            output_dir=temp_dir,
            max_data_points=50
        )
        
        profiler = InferixProfiler(config)
        
        # Test session management
        session_id = profiler.start_session("test_session", tags={"test": True})
        assert session_id == "test_session"
        assert profiler.current_session is not None
        
        # Test stage timing
        with profiler.stage("test_stage_1"):
            time.sleep(0.2)
        
        with profiler.stage("test_stage_2", {"param": "value"}):
            time.sleep(0.1)
        
        # Test custom events
        profiler.add_event("test_event", {"data": 123})
        
        # Test metrics collection
        metrics = profiler.get_current_metrics()
        assert isinstance(metrics, dict)
        assert 'timestamp' in metrics
        
        # End session
        ended_session = profiler.end_session()
        assert ended_session == session_id
        assert profiler.current_session is None
        
        # Check if session was recorded
        assert session_id in profiler.sessions
        session = profiler.sessions[session_id]
        assert session.duration is not None
        assert len(session.stage_timings) >= 2
        assert "test_stage_1" in session.stage_timings
        assert "test_stage_2" in session.stage_timings
        assert len(session.custom_events) >= 1
        
        profiler.cleanup()
        
        print("✓ InferixProfiler tests passed")
        return True


def test_profiling_integration():
    """Test profiling integration utilities."""
    print("Testing ProfilingIntegration...")
    
    # Test disabled profiling
    integration = ProfilingIntegration(None)
    assert not integration.enabled
    
    # Test enabled profiling
    config = ProfilingConfig(enabled=True, real_time_display=False)
    integration = ProfilingIntegration(config)
    assert integration.enabled
    
    # Test session context manager
    with integration.session(prompt="test prompt") as session_id:
        assert session_id is not None
        
        # Test stage context manager
        with integration.stage("test_stage"):
            time.sleep(0.1)
        
        # Test event addition
        integration.add_event("test_event")
        
        # Test metrics
        metrics = integration.get_current_metrics()
        assert isinstance(metrics, dict)
    
    integration.cleanup()
    
    print("✓ ProfilingIntegration tests passed")
    return True


def test_performance_impact():
    """Test performance impact of profiling."""
    print("Testing performance impact...")
    
    # Test without profiling
    start_time = time.time()
    for i in range(100):
        time.sleep(0.001)  # Simulate work
    no_profiling_time = time.time() - start_time
    
    # Test with profiling
    config = ProfilingConfig(
        enabled=True,
        gpu_monitor_interval=0.1,
        cpu_monitor_interval=0.1,
        real_time_display=False,
        generate_final_report=False
    )
    
    profiler = InferixProfiler(config)
    profiler.start_session("perf_test")
    
    start_time = time.time()
    for i in range(100):
        with profiler.stage(f"stage_{i % 10}"):
            time.sleep(0.001)  # Simulate work
    profiling_time = time.time() - start_time
    
    profiler.end_session()
    profiler.cleanup()
    
    # Calculate overhead
    overhead_percent = ((profiling_time - no_profiling_time) / no_profiling_time) * 100
    
    print(f"  Without profiling: {no_profiling_time:.3f}s")
    print(f"  With profiling: {profiling_time:.3f}s")
    print(f"  Overhead: {overhead_percent:.1f}%")
    
    # Expect overhead to be reasonable (< 20%)
    if overhead_percent < 20:
        print("✓ Performance impact test passed")
        return True
    else:
        print(f"✗ Performance impact test failed: {overhead_percent:.1f}% overhead is too high")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Inferix Profiling Module Test Suite")
    print("=" * 60)
    
    tests = [
        test_profiling_config,
        test_monitors,
        test_profiler,
        test_profiling_integration,
        test_performance_impact
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_func in tests:
        print(f"\n{'-' * 40}")
        try:
            if test_func():
                passed_tests += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} failed with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'=' * 60}")
    print(f"Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)