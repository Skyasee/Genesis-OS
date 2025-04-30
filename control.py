import psutil
import platform
import subprocess
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
import json
import asyncio
from datetime import datetime

class SystemInterface:
    """Interface for interacting with system hardware and resources."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.system_info = self._gather_system_info()
        self.monitored_processes = set()
        self.resource_limits = {
            'cpu_percent': 80.0,
            'memory_percent': 75.0,
            'disk_percent': 90.0
        }
    
    def _gather_system_info(self) -> Dict[str, Any]:
        """Gather basic system information."""
        return {
            'platform': platform.system(),
            'platform_release': platform.release(),
            'platform_version': platform.version(),
            'architecture': platform.machine(),
            'processor': platform.processor(),
            'ram': psutil.virtual_memory().total,
            'cpu_cores': psutil.cpu_count()
        }
    
    async def monitor_resources(self) -> Dict[str, float]:
        """Monitor system resource usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'disk_percent': disk.percent,
                'swap_percent': psutil.swap_memory().percent
            }
        except Exception as e:
            self.logger.error(f"Error monitoring resources: {str(e)}")
            return {}

class ProcessController:
    """Controls and manages system processes."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.managed_processes: Dict[int, Dict] = {}
    
    async def start_process(self, command: str, working_dir: Optional[Path] = None) -> Optional[int]:
        """Start a new process."""
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir
            )
            
            self.managed_processes[process.pid] = {
                'command': command,
                'start_time': datetime.now(),
                'working_dir': working_dir
            }
            
            return process.pid
        except Exception as e:
            self.logger.error(f"Failed to start process: {str(e)}")
            return None
    
    async def stop_process(self, pid: int) -> bool:
        """Stop a managed process."""
        try:
            process = psutil.Process(pid)
            process.terminate()
            await asyncio.sleep(3)
            
            if process.is_running():
                process.kill()
            
            self.managed_processes.pop(pid, None)
            return True
        except Exception as e:
            self.logger.error(f"Failed to stop process {pid}: {str(e)}")
            return False

class HardwareController:
    """Controls and manages hardware components."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.system_interface = SystemInterface()
        self.process_controller = ProcessController()
        self.hardware_state = {}
        
    async def initialize(self) -> None:
        """Initialize hardware controller and establish baseline states."""
        try:
            # Get initial hardware state
            self.hardware_state = await self._get_hardware_state()
            
            # Start monitoring
            asyncio.create_task(self._monitor_loop())
            
            self.logger.info("Hardware controller initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize hardware controller: {str(e)}")
    
    async def _get_hardware_state(self) -> Dict[str, Any]:
        """Get current state of hardware components."""
        try:
            battery = psutil.sensors_battery()
            temperatures = psutil.sensors_temperatures()
            fans = psutil.sensors_fans()
            
            return {
                'battery': {
                    'percent': battery.percent if battery else None,
                    'power_plugged': battery.power_plugged if battery else None
                },
                'temperatures': temperatures,
                'fans': fans,
                'resources': await self.system_interface.monitor_resources()
            }
        except Exception as e:
            self.logger.error(f"Error getting hardware state: {str(e)}")
            return {}
    
    async def _monitor_loop(self) -> None:
        """Continuous monitoring loop for hardware status."""
        while True:
            try:
                self.hardware_state = await self._get_hardware_state()
                
                # Check resource limits
                resources = self.hardware_state.get('resources', {})
                for resource, value in resources.items():
                    if value > self.system_interface.resource_limits.get(resource, 100):
                        await self._handle_resource_overflow(resource, value)
                
                await asyncio.sleep(5)  # Monitor every 5 seconds
            except Exception as e:
                self.logger.error(f"Error in monitor loop: {str(e)}")
                await asyncio.sleep(10)  # Wait longer on error
    
    async def _handle_resource_overflow(self, resource: str, value: float) -> None:
        """Handle cases where resource usage exceeds limits."""
        self.logger.warning(f"Resource overflow detected: {resource} at {value}%")
        
        if resource == 'cpu_percent':
            # Find and potentially terminate resource-heavy processes
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
                if proc.info['cpu_percent'] > 50:  # Processes using >50% CPU
                    self.logger.warning(f"High CPU process: {proc.info['name']} ({proc.info['pid']})")
    
    async def execute_command(self, command: str, working_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Execute a system command with resource monitoring."""
        try:
            pid = await self.process_controller.start_process(command, working_dir)
            if pid:
                return {
                    'success': True,
                    'pid': pid,
                    'message': f"Process started successfully"
                }
            return {
                'success': False,
                'message': "Failed to start process"
            }
        except Exception as e:
            return {
                'success': False,
                'message': f"Error executing command: {str(e)}"
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of hardware and running processes."""
        return {
            'hardware_state': self.hardware_state,
            'managed_processes': self.process_controller.managed_processes,
            'system_info': self.system_interface.system_info
        }
    
    async def shutdown(self) -> None:
        """Clean shutdown of hardware controller."""
        try:
            # Stop all managed processes
            for pid in list(self.process_controller.managed_processes.keys()):
                await self.process_controller.stop_process(pid)
            
            self.logger.info("Hardware controller shut down successfully")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}") 