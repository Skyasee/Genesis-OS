import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import logging
from dataclasses import dataclass
from collections import deque
import random

@dataclass
class MemoryState:
    """Represents the current state of the memory system."""
    short_term: torch.Tensor
    long_term: Dict[str, Any]
    attention: torch.Tensor
    context: Dict[str, Any]

class NeuralTuringMachine(nn.Module):
    """Neural Turing Machine for long-term memory management."""
    
    def __init__(self, input_size: int, memory_size: int, num_heads: int = 4):
        super().__init__()
        self.memory_size = memory_size
        self.num_heads = num_heads
        
        # Memory matrix
        self.memory = nn.Parameter(torch.randn(memory_size, input_size))
        
        # Controller network
        self.controller = nn.LSTM(
            input_size=input_size,
            hidden_size=input_size,
            num_layers=2,
            batch_first=True
        )
        
        # Attention mechanisms
        self.query = nn.Linear(input_size, input_size)
        self.key = nn.Linear(input_size, input_size)
        self.value = nn.Linear(input_size, input_size)
        
        # Output projection
        self.output_proj = nn.Linear(input_size * 2, input_size)
    
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Controller forward pass
        controller_out, hidden = self.controller(x, hidden)
        
        # Generate attention queries
        query = self.query(controller_out)
        key = self.key(self.memory)
        value = self.value(self.memory)
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.memory_size)
        attention = torch.softmax(scores, dim=-1)
        
        # Read from memory
        read = torch.matmul(attention, value)
        
        # Combine controller output with memory read
        output = self.output_proj(torch.cat([controller_out, read], dim=-1))
        
        return output, hidden

class MemorySystem:
    """Memory system combining short-term and long-term memory components."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize memory components
        self._init_memory_components()
        
        # Experience replay buffer
        self.experience_buffer = deque(maxlen=10000)
        
        # Current memory state
        self.state = MemoryState(
            short_term=torch.zeros(1, 512).to(self.device),
            long_term={},
            attention=torch.zeros(1, 512).to(self.device),
            context={}
        )
    
    def _init_memory_components(self):
        """Initialize all memory-related components."""
        try:
            # LSTM for short-term memory
            self.lstm = nn.LSTM(
                input_size=512,
                hidden_size=512,
                num_layers=2,
                batch_first=True
            ).to(self.device)
            
            # Transformer for sequence understanding
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=512,
                    nhead=8,
                    dim_feedforward=2048,
                    dropout=0.1
                ),
                num_layers=6
            ).to(self.device)
            
            # Neural Turing Machine for long-term memory
            self.ntm = NeuralTuringMachine(
                input_size=512,
                memory_size=1000,
                num_heads=4
            ).to(self.device)
            
            self.logger.info("Memory components initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing memory components: {str(e)}")
            raise
    
    def process_input(self, input_data: torch.Tensor) -> Dict[str, Any]:
        """Process new input data through the memory system."""
        try:
            # Update short-term memory (LSTM)
            lstm_out, (h_n, c_n) = self.lstm(input_data.unsqueeze(0))
            self.state.short_term = h_n[-1]
            
            # Update transformer context
            transformer_out = self.transformer(input_data.unsqueeze(0))
            self.state.attention = transformer_out.mean(dim=1)
            
            # Update long-term memory (NTM)
            ntm_out, _ = self.ntm(input_data.unsqueeze(0))
            
            # Update context
            self.state.context.update({
                'last_input': input_data.cpu().numpy(),
                'timestamp': self.config.get('current_time', 0)
            })
            
            return {
                'short_term': self.state.short_term,
                'attention': self.state.attention,
                'long_term': ntm_out
            }
        except Exception as e:
            self.logger.error(f"Error processing input: {str(e)}")
            return {}
    
    def store_experience(self, experience: Dict[str, Any]) -> None:
        """Store an experience in the replay buffer."""
        self.experience_buffer.append(experience)
    
    def sample_experiences(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample a batch of experiences from the replay buffer."""
        return random.sample(self.experience_buffer, min(batch_size, len(self.experience_buffer)))
    
    def get_memory_state(self) -> Dict[str, Any]:
        """Get the current state of the memory system."""
        return {
            'short_term': self.state.short_term.cpu().numpy(),
            'attention': self.state.attention.cpu().numpy(),
            'context': self.state.context,
            'buffer_size': len(self.experience_buffer)
        }
    
    def save_state(self, path: str) -> None:
        """Save the current memory system state."""
        state = {
            'lstm': self.lstm.state_dict(),
            'transformer': self.transformer.state_dict(),
            'ntm': self.ntm.state_dict(),
            'experience_buffer': list(self.experience_buffer),
            'config': self.config
        }
        torch.save(state, path)
    
    def load_state(self, path: str) -> None:
        """Load a previously saved memory system state."""
        state = torch.load(path)
        self.lstm.load_state_dict(state['lstm'])
        self.transformer.load_state_dict(state['transformer'])
        self.ntm.load_state_dict(state['ntm'])
        self.experience_buffer = deque(state['experience_buffer'], maxlen=10000)
        self.config = state['config'] 