import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import logging
from dataclasses import dataclass
from collections import deque
import random

class BehaviorCloning(nn.Module):
    """Behavior cloning network for imitation learning."""
    
    def __init__(self, input_size: int, action_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, action_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class Discriminator(nn.Module):
    """Discriminator network for GAIL."""
    
    def __init__(self, input_size: int, action_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size + action_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.network(x)

class ImitationLearning:
    """Imitation learning system combining behavior cloning and GAIL."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize imitation learning components
        self._init_components()
        
        # Expert demonstration buffer
        self.expert_buffer = deque(maxlen=100000)
        
        # Current state
        self.current_state = None
        self.current_action = None
    
    def _init_components(self):
        """Initialize all imitation learning components."""
        try:
            # Behavior cloning network
            self.bc = BehaviorCloning(512, 18).to(self.device)  # 18 common game actions
            
            # GAIL discriminator
            self.discriminator = Discriminator(512, 18).to(self.device)
            
            # Optimizers
            self.bc_optimizer = torch.optim.Adam(self.bc.parameters(), lr=0.001)
            self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0003)
            
            self.logger.info("Imitation learning components initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing imitation learning components: {str(e)}")
            raise
    
    def store_expert_demonstration(self, state: torch.Tensor, action: torch.Tensor) -> None:
        """Store an expert demonstration in the buffer."""
        self.expert_buffer.append({
            'state': state.cpu().numpy(),
            'action': action.cpu().numpy()
        })
    
    def train_behavior_cloning(self, batch_size: int = 32) -> Dict[str, float]:
        """Train the behavior cloning network on expert demonstrations."""
        try:
            if len(self.expert_buffer) < batch_size:
                return {'loss': 0.0}
            
            # Sample batch from expert demonstrations
            batch = random.sample(self.expert_buffer, batch_size)
            states = torch.stack([torch.tensor(exp['state']) for exp in batch]).to(self.device)
            actions = torch.stack([torch.tensor(exp['action']) for exp in batch]).to(self.device)
            
            # Forward pass
            predicted_actions = self.bc(states)
            
            # Compute loss
            loss = F.mse_loss(predicted_actions, actions)
            
            # Update network
            self.bc_optimizer.zero_grad()
            loss.backward()
            self.bc_optimizer.step()
            
            return {'bc_loss': loss.item()}
        except Exception as e:
            self.logger.error(f"Error training behavior cloning: {str(e)}")
            return {'bc_loss': 0.0}
    
    def train_gail(self, states: torch.Tensor, actions: torch.Tensor, 
                  expert_states: torch.Tensor, expert_actions: torch.Tensor) -> Dict[str, float]:
        """Train the GAIL discriminator."""
        try:
            # Get discriminator predictions
            expert_pred = self.discriminator(expert_states, expert_actions)
            agent_pred = self.discriminator(states, actions)
            
            # Compute discriminator loss
            expert_loss = F.binary_cross_entropy(expert_pred, torch.ones_like(expert_pred))
            agent_loss = F.binary_cross_entropy(agent_pred, torch.zeros_like(agent_pred))
            discriminator_loss = expert_loss + agent_loss
            
            # Update discriminator
            self.discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            self.discriminator_optimizer.step()
            
            # Compute generator (agent) loss
            generator_loss = F.binary_cross_entropy(agent_pred, torch.ones_like(agent_pred))
            
            return {
                'discriminator_loss': discriminator_loss.item(),
                'generator_loss': generator_loss.item()
            }
        except Exception as e:
            self.logger.error(f"Error training GAIL: {str(e)}")
            return {'discriminator_loss': 0.0, 'generator_loss': 0.0}
    
    def get_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Get an action using the behavior cloning network."""
        try:
            with torch.no_grad():
                action = self.bc(state)
                confidence = torch.max(F.softmax(action, dim=-1)).item()
                return action, confidence
        except Exception as e:
            self.logger.error(f"Error getting action: {str(e)}")
            return torch.zeros(18).to(self.device), 0.0
    
    def get_reward(self, state: torch.Tensor, action: torch.Tensor) -> float:
        """Get the GAIL reward for a state-action pair."""
        try:
            with torch.no_grad():
                reward = self.discriminator(state, action).item()
                return -np.log(1 - reward)  # Convert to reward
        except Exception as e:
            self.logger.error(f"Error getting reward: {str(e)}")
            return 0.0
    
    def save_state(self, path: str) -> None:
        """Save the current imitation learning state."""
        state = {
            'bc': self.bc.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'expert_buffer': list(self.expert_buffer),
            'config': self.config
        }
        torch.save(state, path)
    
    def load_state(self, path: str) -> None:
        """Load a previously saved imitation learning state."""
        state = torch.load(path)
        self.bc.load_state_dict(state['bc'])
        self.discriminator.load_state_dict(state['discriminator'])
        self.expert_buffer = deque(state['expert_buffer'], maxlen=100000)
        self.config = state['config'] 