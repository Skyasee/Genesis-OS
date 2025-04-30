import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import logging
from dataclasses import dataclass
from collections import deque
import random

class WorldModel(nn.Module):
    """World model for model-based reinforcement learning."""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 512):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_size + action_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # LSTM for temporal dynamics
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, state_size)
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor, 
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Encode state-action pair
        x = torch.cat([state, action], dim=-1)
        encoded = self.encoder(x)
        
        # Process through LSTM
        lstm_out, hidden = self.lstm(encoded.unsqueeze(1), hidden)
        
        # Decode next state
        next_state = self.decoder(lstm_out.squeeze(1))
        
        return next_state, hidden

class MetaLearner(nn.Module):
    """Meta-learning network for fast adaptation to new tasks."""
    
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )
        
        # Meta-optimizer
        self.meta_optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def adapt(self, support_data: List[Dict[str, torch.Tensor]], num_steps: int = 5) -> None:
        """Adapt to new task using support data."""
        for _ in range(num_steps):
            # Sample batch from support data
            batch = random.sample(support_data, min(32, len(support_data)))
            states = torch.stack([item['state'] for item in batch])
            targets = torch.stack([item['target'] for item in batch])
            
            # Compute loss and update
            predictions = self(states)
            loss = F.mse_loss(predictions, targets)
            
            self.meta_optimizer.zero_grad()
            loss.backward()
            self.meta_optimizer.step()

class PlanningSystem:
    """Planning system combining world model and meta-learning."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize planning components
        self._init_components()
        
        # Experience buffer
        self.experience_buffer = deque(maxlen=100000)
        
        # Curriculum state
        self.curriculum_level = 0
        self.curriculum_progress = 0.0
    
    def _init_components(self):
        """Initialize all planning components."""
        try:
            # World model
            self.world_model = WorldModel(512, 18).to(self.device)
            
            # Meta-learner
            self.meta_learner = MetaLearner(512, 18).to(self.device)
            
            # Optimizers
            self.world_model_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=0.001)
            
            self.logger.info("Planning components initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing planning components: {str(e)}")
            raise
    
    def store_experience(self, state: torch.Tensor, action: torch.Tensor, 
                        next_state: torch.Tensor, reward: float) -> None:
        """Store an experience in the buffer."""
        self.experience_buffer.append({
            'state': state.cpu().numpy(),
            'action': action.cpu().numpy(),
            'next_state': next_state.cpu().numpy(),
            'reward': reward
        })
    
    def train_world_model(self, batch_size: int = 32) -> Dict[str, float]:
        """Train the world model on collected experiences."""
        try:
            if len(self.experience_buffer) < batch_size:
                return {'loss': 0.0}
            
            # Sample batch
            batch = random.sample(self.experience_buffer, batch_size)
            states = torch.stack([torch.tensor(exp['state']) for exp in batch]).to(self.device)
            actions = torch.stack([torch.tensor(exp['action']) for exp in batch]).to(self.device)
            next_states = torch.stack([torch.tensor(exp['next_state']) for exp in batch]).to(self.device)
            
            # Predict next states
            predicted_next_states, _ = self.world_model(states, actions)
            
            # Compute loss
            loss = F.mse_loss(predicted_next_states, next_states)
            
            # Update model
            self.world_model_optimizer.zero_grad()
            loss.backward()
            self.world_model_optimizer.step()
            
            return {'world_model_loss': loss.item()}
        except Exception as e:
            self.logger.error(f"Error training world model: {str(e)}")
            return {'world_model_loss': 0.0}
    
    def plan_action(self, state: torch.Tensor, horizon: int = 10) -> Tuple[torch.Tensor, float]:
        """Plan an action using the world model."""
        try:
            with torch.no_grad():
                # Initialize planning
                current_state = state
                best_action = None
                best_value = float('-inf')
                
                # Simulate multiple action sequences
                for _ in range(horizon):
                    # Sample random action
                    action = torch.randn(18).to(self.device)
                    action = F.softmax(action, dim=-1)
                    
                    # Predict next state
                    next_state, _ = self.world_model(current_state, action)
                    
                    # Evaluate using meta-learner
                    value = self.meta_learner(next_state).mean().item()
                    
                    if value > best_value:
                        best_value = value
                        best_action = action
                    
                    current_state = next_state
                
                return best_action, best_value
        except Exception as e:
            self.logger.error(f"Error planning action: {str(e)}")
            return torch.zeros(18).to(self.device), 0.0
    
    def update_curriculum(self, performance: float) -> None:
        """Update the curriculum level based on performance."""
        try:
            # Update progress
            self.curriculum_progress += performance
            
            # Check if ready to advance
            if self.curriculum_progress >= 1.0:
                self.curriculum_level += 1
                self.curriculum_progress = 0.0
                self.logger.info(f"Advanced to curriculum level {self.curriculum_level}")
        except Exception as e:
            self.logger.error(f"Error updating curriculum: {str(e)}")
    
    def get_curriculum_task(self) -> Dict[str, Any]:
        """Get the current curriculum task."""
        return {
            'level': self.curriculum_level,
            'progress': self.curriculum_progress,
            'difficulty': min(1.0, self.curriculum_level / 10.0)
        }
    
    def save_state(self, path: str) -> None:
        """Save the current planning system state."""
        state = {
            'world_model': self.world_model.state_dict(),
            'meta_learner': self.meta_learner.state_dict(),
            'experience_buffer': list(self.experience_buffer),
            'curriculum_level': self.curriculum_level,
            'curriculum_progress': self.curriculum_progress,
            'config': self.config
        }
        torch.save(state, path)
    
    def load_state(self, path: str) -> None:
        """Load a previously saved planning system state."""
        state = torch.load(path)
        self.world_model.load_state_dict(state['world_model'])
        self.meta_learner.load_state_dict(state['meta_learner'])
        self.experience_buffer = deque(state['experience_buffer'], maxlen=100000)
        self.curriculum_level = state['curriculum_level']
        self.curriculum_progress = state['curriculum_progress']
        self.config = state['config'] 