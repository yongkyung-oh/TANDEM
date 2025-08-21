import math
import torch
import torchcde
import torchsde

from torch import Tensor
from .vector_fields import FinalTanhTX


class NeuralTSDE(torch.nn.Module):
    def __init__(self, func, input_channels, hidden_channels, output_channels, initial=True, initial_option=0, attention_option=0):
        super().__init__()
        self.func = func
        self.initial = initial
        self.initial_option = initial_option
        self.attention_option = attention_option
        self.initial_network = torch.nn.Linear(input_channels, hidden_channels)

        self.linear = torch.nn.Sequential(
            torch.nn.Tanh(), torch.nn.Linear(hidden_channels, hidden_channels), 
            torch.nn.ReLU(), torch.nn.Linear(hidden_channels, output_channels), 
        )

        # attention block
        self.positional_encoding = PositionalEncoding(hidden_channels)
        
        self.x_in_1 = torch.nn.Linear(input_channels, hidden_channels)
        self.x_in_2 = torch.nn.Linear(input_channels, hidden_channels)
        # self.x_in_3 = torch.nn.Linear(input_channels, hidden_channels)

        self.attn_0 = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=hidden_channels, nhead=4, dim_feedforward=hidden_channels, batch_first=True), num_layers=2)
        self.attn_1 = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=hidden_channels, nhead=4, dim_feedforward=hidden_channels, batch_first=True), num_layers=2)
        self.attn_2 = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=hidden_channels, nhead=4, dim_feedforward=hidden_channels, batch_first=True), num_layers=2)
        
        # attention weights
        self.attention_weights = torch.nn.Parameter(torch.ones(3))
    
    def forward(self, x, seq, coeffs, times, **kwargs):
        # control module
        self.func.set_X(coeffs, times)
        X = torchcde.CubicSpline(coeffs, times)

        # masked value
        seq_ts = times.repeat(x.shape[0], 1).to(x.device)  # [N,L]
        x = torch.cat([seq_ts.unsqueeze(-1), x], dim=-1)
        xx = torch.where(x == 0, torch.tensor(float('nan')).to(x.device), x)
        # interpolated value and derivative value
        interp = torchcde.CubicSpline(coeffs, times)
        X = torch.stack([interp.evaluate(t) for t in times], dim=-2)
        dX = torch.stack([interp.derivative(t) for t in times], dim=-2)
        
        if self.initial_option == 0: # all zeros
            y0 = torch.zeros(x.shape).to(x.device)
        elif self.initial_option == 1: # all random
            y0 = torch.randn(x.shape).to(x.device)
        elif self.initial_option == 2: # mean imputation 0
            y0 = x
        elif self.initial_option == 3: # interpolation
            y0 = self.func.X.evaluate(times)
        
        y0 = self.initial_network(y0)[:,0,:]

        # Switch default solver
        if 'method' not in kwargs:
            kwargs['method'] = 'srk' # use 'srk' for more accurate solution for SDE 
        if kwargs['method'] == 'srk':
            if 'options' not in kwargs:
                kwargs['options'] = {}
            options = kwargs['options']
            if 'dt' not in options:
                time_diffs = times[1:] - times[:-1]
                options['dt'] = max(time_diffs.min().item(), 1e-3)
        
        # approximation
        if kwargs['method'] == 'euler':
            if 'options' not in kwargs:
                kwargs['options'] = {}
            options = kwargs['options']
            if 'step_size' not in options and 'grid_constructor' not in options:
                time_diffs = times[1:] - times[:-1]
                options['dt'] = max(time_diffs.min().item(), 1e-3)
                
        time_diffs = times[1:] - times[:-1]
        dt = max(time_diffs.min().item(), 1e-3)

        if isinstance(self.func, FinalTanhTX):
            z = torchcde.cdeint(X=self.func.X,
                                func=self.func,
                                z0=y0,
                                t=times,
                                **kwargs)
        else:
            z = torchsde.sdeint(sde=self.func,
                                y0=y0,
                                ts=times,
                                dt=dt,
                                **kwargs)
            z = z.permute(1,0,2) # [N,L,D]

        # select dilated elements for long sequence to aviod OOM
        if len(times) > 2**10:
            idx = select_dilated_elements(times)
        else:
            idx = torch.arange(len(times))

        attention_0 = self.attn_0(self.positional_encoding(z[:,idx,:])).sigmoid()
        attention_1 = self.attn_1(self.positional_encoding(self.x_in_1(x[:,idx,:]))).sigmoid()
        attention_2 = self.attn_2(self.positional_encoding(self.x_in_2(X[:,idx,:]))).sigmoid()

        if self.attention_option == 0:
            pass
        elif self.attention_option == 1:
            z = torch.mul(z[:,idx,:], attention_0)
        elif self.attention_option == 2:
            z = torch.mul(z[:,idx,:], attention_0) + torch.mul(z[:,idx,:], attention_1)
        elif self.attention_option == 3:
            z = torch.mul(z[:,idx,:], attention_0) + torch.mul(z[:,idx,:], attention_2)
        elif self.attention_option == 4:
            z = torch.mul(z[:,idx,:], attention_0) + torch.mul(z[:,idx,:], attention_1) + torch.mul(z[:,idx,:], attention_2)
        elif self.attention_option == 5:
            weights = gumbel_sigmoid(self.attention_weights, tau=1, hard=False)
            z = (weights[0] * torch.mul(z[:,idx,:], attention_0) +
                 weights[1] * torch.mul(z[:,idx,:], attention_1) +
                 weights[2] * torch.mul(z[:,idx,:], attention_2))
        elif self.attention_option == 6:
            weights = gumbel_sigmoid(self.attention_weights, tau=1, hard=True)
            z = (weights[0] * torch.mul(z[:,idx,:], attention_0) +
                 weights[1] * torch.mul(z[:,idx,:], attention_1) +
                 weights[2] * torch.mul(z[:,idx,:], attention_2))
        
        out = self.linear(z)
        return out, z
    

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_seq_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Compute the positional encodings in advance
        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # Register the encodings as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add the positional encodings to the input tensor
        x = x + self.pe[:x.size(0), :].to(x.device)
        return x


def select_dilated_elements(tensor, steps=2):
    if tensor.ndim != 1:
        raise ValueError("Input tensor must be 1-dimensional")
    
    n = len(tensor)
    # Calculate step size to approximately reduce the length by steps
    step = max(1, (n - 1) // ((n // steps) - 1))

    # Select elements using calculated step, ensuring to include the last element
    indices = torch.arange(0, n, step)
    
    # Ensure the last element is included
    if indices[-1] != n - 1:
        indices = torch.cat((indices, torch.tensor([n - 1])))
    
    return indices


def gumbel_sigmoid(logits: Tensor, tau: float = 1.0, hard: bool = False, threshold: float = 0.5) -> Tensor:
    """
    Samples from the Gumbel-Sigmoid distribution and optionally discretizes.
    
    Args:
        logits: `[..., num_features]` unnormalized log probabilities
        tau: non-negative scalar temperature
        hard: if True, the returned samples will be discretized,
              but will be differentiated as if it is the soft sample in autograd
        threshold: threshold for the discretization,
                   values greater than this will be set to 1 and the rest to 0
    
    Returns:
        Sampled tensor of same shape as `logits` from the Gumbel-Sigmoid distribution.
        If hard=True, the returned samples are discretized according to `threshold`,
        otherwise they will be probability distributions.
    """
    if tau <= 0:
        raise ValueError("Temperature tau must be positive")
    
    gumbels = -torch.empty_like(logits).exponential_().log()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.sigmoid()

    if hard:
        # Straight through estimator
        y_hard = (y_soft > threshold).float()
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick
        ret = y_soft
    
    return ret
