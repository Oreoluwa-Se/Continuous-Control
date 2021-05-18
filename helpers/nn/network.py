# import necessary pacakges
import torch
import torch.nn as nn
import torch.nn.functional as F

# initializes the weight and bias data
def layer_init(layer, w_scale=1.0, gain_func="relu"):
    nn.init.orthogonal_(layer.weight.data, gain=nn.init.calculate_gain(gain_func))
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer

def activation_layers(activation):
    if activation.lower() == "relu":
        return F.relu
    if activation.lower() == "tanh":
        return F.tanh

# DDPG Q-Function [action evaluation]
class FCQV(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(32,32), activation="relu", device=torch.device("cpu")):
        # initialize super
        super(FCQV, self).__init__()
        # select activation
        self.activation_fc = activation_layers(activation)
        # build the input layer
        self.input_layer = layer_init(nn.Linear(input_dim, hidden_dims[0]).to(device), gain_func=activation)
        # build hidden layers
        self.hidden_layers = nn.ModuleList()
        for idx in range(len(hidden_dims) - 1):
            in_dim = hidden_dims[idx]
            # in the first layer we increase the dimensions by the output dimension
            if idx == 0:
                in_dim += output_dim
            # initialize layer
            hidden_layer = layer_init(nn.Linear(in_dim, hidden_dims[idx + 1]).to(device), gain_func=activation)
            # append to hidden layer list
            self.hidden_layers.append(hidden_layer)
        # build the output layer - returns [layer_num, 1] q_value
        self.output_layer = layer_init(nn.Linear(hidden_dims[-1], 1).to(device), gain_func=activation)
        # device 
        self.device = device
        
    def _format(self, state, action):
        x, u = state, action
        # set state to right format
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, 
                             device=self.device, 
                             dtype=torch.float32)
            x = x.unsqueeze(0)
        # set actions to right format
        if not isinstance(u, torch.Tensor):
            u = torch.tensor(u, 
                             device=self.device, 
                             dtype=torch.float32)
            u = u.unsqueeze(0)
        # return values
        return x, u

    def forward(self, state, action):
        # check and correct format
        x, u = self._format(state, action)
        # input layer propagation
        x = self.activation_fc(self.input_layer(x))
        # hidden layer propagation
        for i, hidden_layer in enumerate(self.hidden_layers):
            # concatenate the actions to the state in first hidden layer 
            if i == 0:
                x = torch.cat((x,u), dim=1)
            x = self.activation_fc(hidden_layer(x))
        # return
        return self.output_layer(x)
        

# DDPG - Function policy evaluation
class FCDP(nn.Module):
    def __init__(self, input_dim, action_bounds, hidden_dims=(32,32), activation="relu", out_activation="tanh", device=torch.device("cpu")):
        # initialize super
        super(FCDP, self).__init__()
        # select activation
        self.activation_fc = activation_layers(activation)
        self.out_activation_fc = activation_layers(out_activation)
        # action bounds - represent the maximum and minimum values of possible actions. Used for resizing from [-1, 1] range
        self.env_min, self.env_max = action_bounds
        # build the input layer
        self.input_layer = layer_init(nn.Linear(input_dim, hidden_dims[0]).to(device), gain_func=activation)
        # build hidden layers
        self.hidden_layers = nn.ModuleList()
        for idx in range(len(hidden_dims) - 1):
            # initialize layer
            hidden_layer = layer_init(nn.Linear(hidden_dims[idx], hidden_dims[idx + 1]).to(device), gain_func=activation)
            # append to hidden layer list
            self.hidden_layers.append(hidden_layer)
        # build the output layer - returns [layer_num, 1] q_value
        self.output_layer = layer_init(nn.Linear(hidden_dims[-1], len(self.env_max)).to(device), gain_func=out_activation)
        # device 
        self.device = device
        # convert env_min and max to torch tensors
        self.env_min = torch.tensor(self.env_min,
                                    device=self.device,
                                    dtype=torch.float32)
        self.env_max = torch.tensor(self.env_max,
                                    device=self.device,
                                    dtype=torch.float32)
        # create array tensor to represent activated extremes of tanh
        self.nn_min = self.out_activation_fc(torch.Tensor([float("-inf")])).to(self.device)
        self.nn_max = self.out_activation_fc(torch.Tensor([float("inf")])).to(self.device)
        # rescale function
        self.rescale_fn = lambda x: ((x - self.nn_min) * (self.env_max - self.env_min) / (self.nn_max - self.nn_min)) + self.env_min 
        
    def _format(self, state):
        x = state
        # set state to right format
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, 
                             device=self.device, 
                             dtype=torch.float32)
            x = x.unsqueeze(0)
        # return values
        return x

    def forward(self, state):
        # check and correct format
        x = self._format(state)
        # input layer propagation
        x = self.activation_fc(self.input_layer(x))
        # hidden layer propagation
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        # output layer
        x = self.output_layer(x)
        # output layer activation
        x = self.out_activation_fc(x)
        # return rescaled action from -1 to 1 range to specified range in the environment
        return self.rescale_fn(x)

    