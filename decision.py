import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import logging
from dataclasses import dataclass
from collections import deque
import random

@dataclass
class Action:
    """Represents a game action with its parameters."""
    action_type: str
    parameters: Dict[str, Any]
    confidence: float

class DQN(nn.Module):
    """Deep Q-Network for discrete action spaces."""
    
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class PPOActor(nn.Module):
    """Actor network for PPO algorithm."""
    
    def __init__(self, input_size: int, action_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_size),
            nn.Tanh()  # Output actions in [-1, 1]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class PPOCritic(nn.Module):
    """Critic network for PPO algorithm."""
    
    def __init__(self, input_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class SAC(nn.Module):
    """Soft Actor-Critic for continuous action spaces."""
    
    def __init__(self, input_size: int, action_size: int):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_size * 2)  # Mean and log_std
        )
        
        self.critic1 = nn.Sequential(
            nn.Linear(input_size + action_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        self.critic2 = nn.Sequential(
            nn.Linear(input_size + action_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Actor forward pass
        actor_out = self.actor(state)
        mean, log_std = actor_out.chunk(2, dim=-1)
        std = log_std.exp()
        
        if action is None:
            # Sample action
            noise = torch.randn_like(mean)
            action = mean + std * noise
        
        # Critic forward pass
        state_action = torch.cat([state, action], dim=-1)
        q1 = self.critic1(state_action)
        q2 = self.critic2(state_action)
        
        return action, q1, q2

class DecisionEngine:
    """Decision engine combining multiple RL algorithms."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize decision components
        self._init_decision_components()
        
        # Experience replay buffer
        self.replay_buffer = deque(maxlen=100000)
        
        # Current state
        self.current_state = None
        self.current_action = None
    
    def _init_decision_components(self):
        """Initialize all decision-making components."""
        try:
            # DQN for discrete actions
            self.dqn = DQN(512, 18).to(self.device)  # 18 common game actions
            
            # PPO components
            self.ppo_actor = PPOActor(512, 8).to(self.device)  # 8 continuous actions
            self.ppo_critic = PPOCritic(512).to(self.device)
            
            # SAC for complex continuous actions
            self.sac = SAC(512, 8).to(self.device)
            
            # Optimizers
            self.dqn_optimizer = torch.optim.Adam(self.dqn.parameters(), lr=0.001)
            self.ppo_optimizer = torch.optim.Adam(list(self.ppo_actor.parameters()) + 
                                                list(self.ppo_critic.parameters()), lr=0.0003)
            self.sac_optimizer = torch.optim.Adam(list(self.sac.parameters()), lr=0.0003)
            
            self.logger.info("Decision components initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing decision components: {str(e)}")
            raise
    
    def select_action(self, state: torch.Tensor, action_space: str = 'discrete') -> Action:
        """Select an action based on the current state."""
        try:
            with torch.no_grad():
                if action_space == 'discrete':
                    # Use DQN for discrete actions
                    q_values = self.dqn(state)
                    action_idx = q_values.argmax().item()
                    confidence = F.softmax(q_values, dim=-1)[action_idx].item()
                    
                    return Action(
                        action_type='discrete',
                        parameters={'action_idx': action_idx},
                        confidence=confidence
                    )
                
                elif action_space == 'continuous':
                    # Use PPO for continuous actions
                    action = self.ppo_actor(state)
                    value = self.ppo_critic(state)
                    
                    return Action(
                        action_type='continuous',
                        parameters={'action': action.cpu().numpy()},
                        confidence=value.item()
                    )
                
                else:  # complex
                    # Use SAC for complex continuous actions
                    action, q1, q2 = self.sac(state)
                    q_value = torch.min(q1, q2)
                    
                    return Action(
                        action_type='complex',
                        parameters={'action': action.cpu().numpy()},
                        confidence=q_value.item()
                    )
        except Exception as e:
            self.logger.error(f"Error selecting action: {str(e)}")
            return Action(action_type='none', parameters={}, confidence=0.0)
    
    def update(self, batch: List[Dict[str, Any]], algorithm: str = 'dqn') -> Dict[str, float]:
        """Update the decision models using a batch of experiences."""
        try:
            if algorithm == 'dqn':
                return self._update_dqn(batch)
            elif algorithm == 'ppo':
                return self._update_ppo(batch)
            elif algorithm == 'sac':
                return self._update_sac(batch)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
        except Exception as e:
            self.logger.error(f"Error updating models: {str(e)}")
            return {'loss': 0.0}
    
    def _update_dqn(self, batch: List[Dict[str, Any]]) -> Dict[str, float]:
        """Update DQN using experience replay."""
        states = torch.stack([torch.tensor(exp['state']) for exp in batch]).to(self.device)
        actions = torch.tensor([exp['action'] for exp in batch]).to(self.device)
        rewards = torch.tensor([exp['reward'] for exp in batch]).to(self.device)
        next_states = torch.stack([torch.tensor(exp['next_state']) for exp in batch]).to(self.device)
        dones = torch.tensor([exp['done'] for exp in batch]).to(self.device)
        
        # Compute current Q values
        current_q = self.dqn(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q values
        with torch.no_grad():
            next_q = self.dqn(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * 0.99 * next_q
        
        # Compute loss and update
        loss = F.smooth_l1_loss(current_q.squeeze(), target_q)
        self.dqn_optimizer.zero_grad()
        loss.backward()
        self.dqn_optimizer.step()
        
        return {'loss': loss.item()}
    
    def _update_ppo(self, batch: List[Dict[str, Any]]) -> Dict[str, float]:
        """Update PPO models."""
        states = torch.stack([torch.tensor(exp['state']) for exp in batch]).to(self.device)
        actions = torch.stack([torch.tensor(exp['action']) for exp in batch]).to(self.device)
        rewards = torch.tensor([exp['reward'] for exp in batch]).to(self.device)
        old_log_probs = torch.tensor([exp['log_prob'] for exp in batch]).to(self.device)
        
        # Compute new action probabilities and values
        new_actions = self.ppo_actor(states)
        values = self.ppo_critic(states)
        
        # Compute PPO loss
        ratio = torch.exp(new_actions - old_log_probs)
        surr1 = ratio * rewards
        surr2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * rewards
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = F.mse_loss(values.squeeze(), rewards)
        
        # Update models
        loss = actor_loss + 0.5 * critic_loss
        self.ppo_optimizer.zero_grad()
        loss.backward()
        self.ppo_optimizer.step()
        
        return {'actor_loss': actor_loss.item(), 'critic_loss': critic_loss.item()}
    
    def _update_sac(self, batch: List[Dict[str, Any]]) -> Dict[str, float]:
        """Update SAC model."""
        states = torch.stack([torch.tensor(exp['state']) for exp in batch]).to(self.device)
        actions = torch.stack([torch.tensor(exp['action']) for exp in batch]).to(self.device)
        rewards = torch.tensor([exp['reward'] for exp in batch]).to(self.device)
        next_states = torch.stack([torch.tensor(exp['next_state']) for exp in batch]).to(self.device)
        dones = torch.tensor([exp['done'] for exp in batch]).to(self.device)
        
        # Update SAC
        next_actions, next_q1, next_q2 = self.sac(next_states)
        next_q = torch.min(next_q1, next_q2)
        target_q = rewards + (1 - dones) * 0.99 * next_q
        
        current_actions, current_q1, current_q2 = self.sac(states, actions)
        current_q = torch.min(current_q1, current_q2)
        
        loss = F.mse_loss(current_q.squeeze(), target_q.detach())
        
        self.sac_optimizer.zero_grad()
        loss.backward()
        self.sac_optimizer.step()
        
        return {'loss': loss.item()}
    
    def save_state(self, path: str) -> None:
        """Save the current decision models state."""
        state = {
            'dqn': self.dqn.state_dict(),
            'ppo_actor': self.ppo_actor.state_dict(),
            'ppo_critic': self.ppo_critic.state_dict(),
            'sac': self.sac.state_dict(),
            'config': self.config
        }
        torch.save(state, path)
    
    def load_state(self, path: str) -> None:
        """Load a previously saved decision models state."""
        state = torch.load(path)
        self.dqn.load_state_dict(state['dqn'])
        self.ppo_actor.load_state_dict(state['ppo_actor'])
        self.ppo_critic.load_state_dict(state['ppo_critic'])
        self.sac.load_state_dict(state['sac'])
        self.config = state['config'] 