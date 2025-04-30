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
        try:
            import psutil
            return psutil.cpu_percent() / 100.0
        except ImportError:
            self.logger.warning("psutil not available, using default CPU usage")
            return 0.0
    
    def _get_memory_usage(self) -> float:
        """Get memory usage percentage."""
        try:
            import psutil
            return psutil.virtual_memory().percent / 100.0
        except ImportError:
            self.logger.warning("psutil not available, using default memory usage")
            return 0.0
    
    def _get_gpu_usage(self) -> float:
        """Get GPU usage percentage."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return info.used / info.total
        except (ImportError, Exception) as e:
            self.logger.warning(f"GPU monitoring not available: {str(e)}")
            return 0.0
    
    def _optimize_cpu_usage(self) -> None:
        """Optimize CPU usage."""
        try:
            # Reduce batch sizes for neural networks
            # Adjust thread count for parallel operations
            self.logger.info("Optimizing CPU usage")
        except Exception as e:
            self.logger.error(f"Error optimizing CPU usage: {str(e)}")
    
    def _optimize_memory_usage(self) -> None:
        """Optimize memory usage."""
        try:
            # Clear unnecessary caches
            # Reduce buffer sizes
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            self.logger.info("Optimizing memory usage")
        except Exception as e:
            self.logger.error(f"Error optimizing memory usage: {str(e)}")
    
    def _optimize_gpu_usage(self) -> None:
        """Optimize GPU usage."""
        try:
            # Reduce model precision
            # Adjust batch sizes
            self.logger.info("Optimizing GPU usage")
        except Exception as e:
            self.logger.error(f"Error optimizing GPU usage: {str(e)}")

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
        try:
            task_type = task.get('type', 'unknown')
            
            if task_type == 'vision':
                return await self._run_vision_task(task)
            elif task_type == 'decision':
                return await self._run_decision_task(task)
            elif task_type == 'learning':
                return await self._run_learning_task(task)
            elif task_type == 'evolution':
                return await self._run_evolution_task(task)
            else:
                return {'status': 'error', 'error': f"Unknown task type: {task_type}"}
        except Exception as e:
            self.logger.error(f"Error running task: {str(e)}")
            return {'status': 'error', 'error': str(e)}
    
    async def _run_vision_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Run a vision-related task."""
        # Implementation for vision tasks
        return {'status': 'success', 'type': 'vision'}
    
    async def _run_decision_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Run a decision-related task."""
        # Implementation for decision tasks
        return {'status': 'success', 'type': 'decision'}
    
    async def _run_learning_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Run a learning-related task."""
        # Implementation for learning tasks
        return {'status': 'success', 'type': 'learning'}
    
    async def _run_evolution_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Run an evolution-related task."""
        # Implementation for evolution tasks
        return {'status': 'success', 'type': 'evolution'}

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
        try:
            # Calculate exploration rate based on task success
            task_success = experience.get('result', {}).get('status') == 'success'
            current_rate = float(self.current_phase == 'exploration')
            
            # Adjust exploration rate based on success
            if task_success:
                # Reduce exploration as we succeed more
                new_rate = max(0.1, current_rate - 0.05)
            else:
                # Increase exploration when we fail
                new_rate = min(1.0, current_rate + 0.1)
            
            # Update phase
            if new_rate < 0.3:
                self.current_phase = 'exploitation'
            else:
                self.current_phase = 'exploration'
            
            return new_rate
        except Exception as e:
            self.logger.error(f"Error updating exploration: {str(e)}")
            return 0.5
    
    def _update_imitation(self, experience: Dict[str, Any]) -> float:
        """Update imitation learning."""
        try:
            # Calculate imitation progress based on expert demonstrations
            expert_actions = experience.get('expert_actions', [])
            agent_actions = experience.get('agent_actions', [])
            
            if not expert_actions or not agent_actions:
                return 0.0
            
            # Calculate similarity between expert and agent actions
            similarities = []
            for exp_action, agent_action in zip(expert_actions, agent_actions):
                if isinstance(exp_action, torch.Tensor) and isinstance(agent_action, torch.Tensor):
                    similarity = torch.cosine_similarity(exp_action, agent_action, dim=0)
                    similarities.append(similarity.item())
            
            return np.mean(similarities) if similarities else 0.0
        except Exception as e:
            self.logger.error(f"Error updating imitation: {str(e)}")
            return 0.0
    
    def _update_planning(self, experience: Dict[str, Any]) -> float:
        """Update planning system."""
        try:
            # Calculate planning accuracy based on predicted vs actual outcomes
            predicted_outcomes = experience.get('predicted_outcomes', [])
            actual_outcomes = experience.get('actual_outcomes', [])
            
            if not predicted_outcomes or not actual_outcomes:
                return 0.0
            
            # Calculate accuracy
            correct = sum(1 for p, a in zip(predicted_outcomes, actual_outcomes) if p == a)
            return correct / len(predicted_outcomes)
        except Exception as e:
            self.logger.error(f"Error updating planning: {str(e)}")
            return 0.0
    
    def _update_meta_learning(self, experience: Dict[str, Any]) -> float:
        """Update meta-learning system."""
        try:
            # Calculate meta-learning score based on adaptation speed
            previous_performance = experience.get('previous_performance', 0.0)
            current_performance = experience.get('current_performance', 0.0)
            
            if previous_performance == 0.0:
                return 0.0
            
            # Calculate improvement ratio
            improvement = (current_performance - previous_performance) / max(0.01, previous_performance)
            return max(0.0, min(1.0, improvement))
        except Exception as e:
            self.logger.error(f"Error updating meta-learning: {str(e)}")
            return 0.0

class GenesisOS:
    """Main Genesis OS class integrating all AI components."""
    
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
    
    def _update_learning(self, state: Dict[str, Any], action: Any, result: Dict[str, Any]):
        """Update learning based on experience."""
        try:
            # Update various learning components
            self.decision.update([{
                'state': state,
                'action': action,
                'reward': result.get('reward', 0.0),
                'next_state': result.get('next_state', state),
                'done': result.get('done', False)
            }])
            
            # Update imitation learning if expert demonstration available
            if 'expert_action' in result:
                self.imitation.store_expert_demonstration(
                    torch.tensor(state).to(self.device),
                    torch.tensor(result['expert_action']).to(self.device)
                )
            
            # Update planning system
            self.planning.store_experience(
                torch.tensor(state).to(self.device),
                torch.tensor(action).to(self.device),
                torch.tensor(result.get('next_state', state)).to(self.device),
                result.get('reward', 0.0)
            )
            
            # Train world model
            self.planning.train_world_model()
            
            return {
                'decision_loss': 0.0,  # Would be updated by decision engine
                'imitation_loss': 0.0,  # Would be updated by imitation learning
                'planning_loss': 0.0   # Would be updated by planning system
            }
        except Exception as e:
            self.logger.error(f"Error updating learning: {str(e)}")
            return {}
    
    def _update_metrics(self, result: Dict[str, Any], 
                       learning_metrics: Dict[str, float] = None,
                       evolution_result: Dict[str, Any] = None):
        """Update system performance metrics."""
        try:
            # Basic metrics
            metrics = {
                'task_success': result.get('status') == 'success',
                'timestamp': datetime.now()
            }
            
            # Add learning metrics if available
            if learning_metrics:
                metrics['learning_progress'] = np.mean(list(learning_metrics.values()))
            
            # Add evolution metrics if available
            if evolution_result:
                metrics['evolution_progress'] = evolution_result.get('performance_score', 0.0)
                self.evolution_history.append(evolution_result)
            
            # Update performance metrics
            self.performance_metrics = metrics
            
            # Log metrics
            self.logger.info(f"Performance metrics updated: {metrics}")
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