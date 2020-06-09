from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        """
        Args:
            in_dim:
            out_dim:
        """
        super(ResidualBlock, self).__init__()
        # if input and output dimensions match, use residual connection
        self.residual_connection = True if in_dim == out_dim else False
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = nn.PReLU()

    def forward(self, x):
        y = self.activation(self.linear(x))
        if self.residual_connection:
            y = x + y
        return y


class ResFNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims):
        """
        Feedforward neural network with residual connection.

        Args:
            input_dim: integer, specifies input dimension of the neural network
            output_dim: integer, specifies output dimension of the neural network
            hidden_dims: list of integers, specifies the hidden dimensions of each layer.
                in above definition L = len(hidden_dims) since the last hidden layer is followed by an output layer
        """
        super(ResFNN, self).__init__()
        blocks = list()
        self.input_dim = input_dim
        input_dim_block = input_dim
        for hidden_dim in hidden_dims:
            blocks.append(ResidualBlock(input_dim_block, hidden_dim))  # here the hidden layer is specified
            # blocks.append(nn.PReLU()) # we fix this activation function, because they work well in practice - not smooth.
            input_dim_block = hidden_dim
        blocks.append(nn.Linear(input_dim_block, output_dim))
        self.network = nn.Sequential(*blocks)
        self.blocks = blocks

    def forward(self, x, domain=0):
        if x.shape[2] > 0:
            # flatten the processes dimension
            x = x.reshape(x.shape[0], -1)
        out = self.network(x)
        return out
