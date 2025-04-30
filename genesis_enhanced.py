import torch
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
import asyncio
from datetime import datetime
import numpy as np
from dataclasses import dataclass
from collections import deque

from .brain import Brain
from .vision import VisionPerception
from .memory import MemorySystem
from .decision import DecisionEngine
from .imitation import ImitationLearning
from .planning import PlanningSystem
from .control import HardwareController
from .prism_x import PrismX
from .evo_forge import EvoForge

@dataclass
class SystemMetrics:
    """Comprehensive system performance metrics."""
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    task_success_rate: float
    learning_progress: float
    adaptation_speed: float
    stability_score: float
    energy_efficiency: float

class ResourceManager:
    """Manages system resources and optimization."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics_history = deque(maxlen=1000)
        self.optimization_threshold = 0.8
    
    def monitor_resources(self) -> SystemMetrics:
        """Monitor system resource usage."""
        try:
            # Get resource usage metrics
            cpu_usage = self._get_cpu_usage()
            memory_usage = self._get_memory_usage()
            gpu_usage = self._get_gpu_usage()
            
            return SystemMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                gpu_usage=gpu_usage,
                task_success_rate=0.0,  # Updated by task manager
                learning_progress=0.0,  # Updated by learning manager
                adaptation_speed=0.0,   # Updated by evolution manager
                stability_score=0.0,    # Updated by stability monitor
                energy_efficiency=0.0   # Calculated from resource usage
            )
        except Exception as e:
            self.logger.error(f"Error monitoring resources: {str(e)}")
            return SystemMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    def optimize_resources(self, metrics: SystemMetrics) -> None:
        """Optimize resource usage based on metrics."""
        try:
            if metrics.cpu_usage > self.optimization_threshold:
                self._optimize_cpu_usage()
            if metrics.memory_usage > self.optimization_threshold:
                self._optimize_memory_usage()
            if metrics.gpu_usage > self.optimization_threshold:
                self._optimize_gpu_usage()
        except Exception as e:
            self.logger.error(f"Error optimizing resources: {str(e)}")
    
    def _get_cpu_usage(self) -> float:
        """Get CPU usage percentage."""
        # Implementation depends on OS
        return 0.0
    
    def _get_memory_usage(self) -> float:
        """Get memory usage percentage."""
        # Implementation depends on OS
        return 0.0
    
    def _get_gpu_usage(self) -> float:
        """Get GPU usage percentage."""
        # Implementation depends on GPU monitoring tools
        return 0.0
    
    def _optimize_cpu_usage(self) -> None:
        """Optimize CPU usage."""
        # Implementation for CPU optimization
        pass
    
    def _optimize_memory_usage(self) -> None:
        """Optimize memory usage."""
        # Implementation for memory optimization
        pass
    
    def _optimize_gpu_usage(self) -> None:
        """Optimize GPU usage."""
        # Implementation for GPU optimization
        pass

class TaskManager:
    """Manages task execution and scheduling."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.task_queue = asyncio.Queue()
        self.task_history = deque(maxlen=1000)
        self.current_task = None
    
    async def add_task(self, task: Dict[str, Any]) -> None:
        """Add a task to the queue."""
        try:
            await self.task_queue.put(task)
            self.logger.info(f"Task added to queue: {task['name']}")
        except Exception as e:
            self.logger.error(f"Error adding task: {str(e)}")
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single task."""
        try:
            self.current_task = task
            result = await self._run_task(task)
            self.task_history.append({
                'task': task,
                'result': result,
                'timestamp': datetime.now()
            })
            return result
        except Exception as e:
            self.logger.error(f"Error executing task: {str(e)}")
            return {'status': 'error', 'error': str(e)}
    
    async def _run_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Run a task with proper resource management."""
        # Implementation for task execution
        return {'status': 'success'}

class LearningManager:
    """Manages learning processes and optimization."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.learning_history = deque(maxlen=1000)
        self.current_phase = 'exploration'
    
    def update_learning(self, experience: Dict[str, Any]) -> Dict[str, float]:
        """Update learning based on new experience."""
        try:
            # Update various learning components
            metrics = {
                'exploration_rate': self._update_exploration(experience),
                'imitation_progress': self._update_imitation(experience),
                'planning_accuracy': self._update_planning(experience),
                'meta_learning_score': self._update_meta_learning(experience)
            }
            
            self.learning_history.append({
                'experience': experience,
                'metrics': metrics,
                'timestamp': datetime.now()
            })
            
            return metrics
        except Exception as e:
            self.logger.error(f"Error updating learning: {str(e)}")
            return {}
    
    def _update_exploration(self, experience: Dict[str, Any]) -> float:
        """Update exploration strategy."""
        # Implementation for exploration update
        return 0.0
    
    def _update_imitation(self, experience: Dict[str, Any]) -> float:
        """Update imitation learning."""
        # Implementation for imitation update
        return 0.0
    
    def _update_planning(self, experience: Dict[str, Any]) -> float:
        """Update planning system."""
        # Implementation for planning update
        return 0.0
    
    def _update_meta_learning(self, experience: Dict[str, Any]) -> float:
        """Update meta-learning system."""
        # Implementation for meta-learning update
        return 0.0

class GenesisOSEnhanced:
    """Enhanced Genesis OS with advanced features."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self._init_components()
        
        # Initialize managers
        self.resource_manager = ResourceManager()
        self.task_manager = TaskManager()
        self.learning_manager = LearningManager()
        
        # State tracking
        self.is_running = False
        self.performance_metrics = {}
        self.evolution_history = []
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        
        return {
            'vision': {
                'frame_buffer_size': 30,
                'detection_threshold': 0.5
            },
            'memory': {
                'buffer_size': 100000,
                'hidden_size': 512
            },
            'decision': {
                'batch_size': 32,
                'learning_rate': 0.001
            },
            'imitation': {
                'expert_buffer_size': 100000,
                'bc_learning_rate': 0.001
            },
            'planning': {
                'horizon': 10,
                'curriculum_levels': 10
            },
            'evolution': {
                'population_size': 5,
                'mutation_rate': 0.1
            },
            'resource_management': {
                'optimization_threshold': 0.8,
                'monitoring_interval': 1.0
            }
        }
    
    def _init_components(self):
        """Initialize all Genesis OS components."""
        try:
            # Core brain
            self.brain = Brain(self.config.get('brain', {}))
            
            # Vision system
            self.vision = VisionPerception(self.config.get('vision', {}))
            
            # Memory system
            self.memory = MemorySystem(self.config.get('memory', {}))
            
            # Decision engine
            self.decision = DecisionEngine(self.config.get('decision', {}))
            
            # Imitation learning
            self.imitation = ImitationLearning(self.config.get('imitation', {}))
            
            # Planning system
            self.planning = PlanningSystem(self.config.get('planning', {}))
            
            # Hardware controller
            self.hardware = HardwareController()
            
            # PRISM-X integration
            self.prism_x = PrismX(self.config.get('prism_x', {}))
            
            # EVO-Forge integration
            self.evo_forge = EvoForge(self.config.get('evolution', {}))
            
            self.logger.info("All Genesis OS components initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing Genesis OS components: {str(e)}")
            raise
    
    async def start(self):
        """Start the Genesis OS."""
        try:
            self.is_running = True
            self.logger.info("Starting Genesis OS...")
            
            # Initialize hardware controller
            await self.hardware.initialize()
            
            # Start main loop
            await self._main_loop()
        except Exception as e:
            self.logger.error(f"Error starting Genesis OS: {str(e)}")
            self.is_running = False
    
    async def stop(self):
        """Stop the Genesis OS."""
        try:
            self.is_running = False
            self.logger.info("Stopping Genesis OS...")
            
            # Save states
            self._save_states()
            
            # Cleanup hardware
            await self.hardware.cleanup()
        except Exception as e:
            self.logger.error(f"Error stopping Genesis OS: {str(e)}")
    
    async def _main_loop(self):
        """Main execution loop."""
        try:
            while self.is_running:
                # Monitor resources
                metrics = self.resource_manager.monitor_resources()
                self.resource_manager.optimize_resources(metrics)
                
                # Process task queue
                if not self.task_manager.task_queue.empty():
                    task = await self.task_manager.task_queue.get()
                    result = await self.task_manager.execute_task(task)
                    
                    # Update learning
                    learning_metrics = self.learning_manager.update_learning({
                        'task': task,
                        'result': result,
                        'metrics': metrics
                    })
                    
                    # Update evolution
                    evolution_result = self.evo_forge.evolve({
                        'task': task,
                        'result': result,
                        'learning_metrics': learning_metrics,
                        'system_metrics': metrics
                    })
                    
                    # Update performance metrics
                    self._update_metrics(result, learning_metrics, evolution_result)
                
                # Sleep to prevent CPU overload
                await asyncio.sleep(0.1)
        except Exception as e:
            self.logger.error(f"Error in main loop: {str(e)}")
            self.is_running = False
    
    def _update_metrics(self, result: Dict[str, Any], 
                       learning_metrics: Dict[str, float],
                       evolution_result: Dict[str, Any]) -> None:
        """Update system performance metrics."""
        try:
            self.performance_metrics = {
                'task_success': result.get('status') == 'success',
                'learning_progress': np.mean(list(learning_metrics.values())),
                'evolution_progress': evolution_result.get('performance_score', 0.0),
                'timestamp': datetime.now()
            }
        except Exception as e:
            self.logger.error(f"Error updating metrics: {str(e)}")
    
    def _save_states(self):
        """Save the current state of all components."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = Path("states") / timestamp
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save component states
            self.brain.save_state(str(save_dir / "brain.pt"))
            self.vision.save_state(str(save_dir / "vision.pt"))
            self.memory.save_state(str(save_dir / "memory.pt"))
            self.decision.save_state(str(save_dir / "decision.pt"))
            self.imitation.save_state(str(save_dir / "imitation.pt"))
            self.planning.save_state(str(save_dir / "planning.pt"))
            self.prism_x.save_state(str(save_dir / "prism_x.pt"))
            self.evo_forge.save_state(str(save_dir / "evo_forge.pt"))
            
            # Save system state
            system_state = {
                'performance_metrics': self.performance_metrics,
                'evolution_history': self.evolution_history,
                'config': self.config
            }
            
            with open(save_dir / "system_state.json", 'w') as f:
                json.dump(system_state, f)
            
            self.logger.info(f"States saved to {save_dir}")
        except Exception as e:
            self.logger.error(f"Error saving states: {str(e)}")
    
    def load_states(self, path: str):
        """Load previously saved states."""
        try:
            load_dir = Path(path)
            
            # Load component states
            self.brain.load_state(str(load_dir / "brain.pt"))
            self.vision.load_state(str(load_dir / "vision.pt"))
            self.memory.load_state(str(load_dir / "memory.pt"))
            self.decision.load_state(str(load_dir / "decision.pt"))
            self.imitation.load_state(str(load_dir / "imitation.pt"))
            self.planning.load_state(str(load_dir / "planning.pt"))
            self.prism_x.load_state(str(load_dir / "prism_x.pt"))
            self.evo_forge.load_state(str(load_dir / "evo_forge.pt"))
            
            # Load system state
            with open(load_dir / "system_state.json", 'r') as f:
                system_state = json.load(f)
            
            self.performance_metrics = system_state['performance_metrics']
            self.evolution_history = system_state['evolution_history']
            self.config = system_state['config']
            
            self.logger.info(f"States loaded from {load_dir}")
        except Exception as e:
            self.logger.error(f"Error loading states: {str(e)}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status."""
        try:
            return {
                'is_running': self.is_running,
                'current_task': self.task_manager.current_task,
                'performance_metrics': self.performance_metrics,
                'resource_metrics': self.resource_manager.monitor_resources(),
                'learning_metrics': self.learning_manager.learning_history[-1] if self.learning_manager.learning_history else {},
                'evolution_metrics': self.evo_forge.evolution_manager.performance_monitor.calculate_metrics()
            }
        except Exception as e:
            self.logger.error(f"Error getting status: {str(e)}")
            return {} 