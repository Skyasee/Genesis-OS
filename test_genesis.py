import unittest
import asyncio
from genius_ai.core.genesis import GenesisOS

class TestGenesisOS(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test."""
        self.genesis = GenesisOS()
        
    def test_initialization(self):
        """Test if GenesisOS initializes correctly."""
        self.assertIsNotNone(self.genesis)
        self.assertIsNotNone(self.genesis.brain)
        self.assertIsNotNone(self.genesis.vision)
        self.assertIsNotNone(self.genesis.memory)
        self.assertIsNotNone(self.genesis.decision)
        self.assertIsNotNone(self.genesis.imitation)
        self.assertIsNotNone(self.genesis.planning)
        self.assertIsNotNone(self.genesis.control)
        
    def test_resource_manager(self):
        """Test if ResourceManager initializes and functions correctly."""
        self.assertIsNotNone(self.genesis.resource_manager)
        metrics = self.genesis.resource_manager.monitor_resources()
        self.assertIsNotNone(metrics)
        self.assertIsInstance(metrics.cpu_usage, float)
        self.assertIsInstance(metrics.memory_usage, float)
        self.assertIsInstance(metrics.gpu_usage, float)
        
    def test_task_manager(self):
        """Test if TaskManager initializes and functions correctly."""
        self.assertIsNotNone(self.genesis.task_manager)
        self.assertEqual(len(self.genesis.task_manager.task_queue), 0)
        
    def test_learning_manager(self):
        """Test if LearningManager initializes and functions correctly."""
        self.assertIsNotNone(self.genesis.learning_manager)
        self.assertEqual(len(self.genesis.learning_manager.learning_history), 0)
        
    @asyncio.coroutine
    async def test_start_stop(self):
        """Test if GenesisOS starts and stops correctly."""
        # Start the system
        await self.genesis.start()
        self.assertTrue(self.genesis.running)
        
        # Stop the system
        await self.genesis.stop()
        self.assertFalse(self.genesis.running)
        
    def test_state_management(self):
        """Test if state saving and loading works correctly."""
        # Save state
        state = self.genesis.save_state()
        self.assertIsNotNone(state)
        
        # Load state
        self.genesis.load_state(state)
        self.assertIsNotNone(self.genesis.brain)
        self.assertIsNotNone(self.genesis.vision)
        self.assertIsNotNone(self.genesis.memory)
        
    def test_config_loading(self):
        """Test if configuration loading works correctly."""
        config = {
            'vision': {
                'frame_buffer_size': 60,
                'detection_threshold': 0.7
            },
            'memory': {
                'buffer_size': 200000,
                'hidden_size': 1024
            }
        }
        
        genesis = GenesisOS(config=config)
        self.assertEqual(genesis.config['vision']['frame_buffer_size'], 60)
        self.assertEqual(genesis.config['memory']['buffer_size'], 200000)
        
if __name__ == '__main__':
    unittest.main() 