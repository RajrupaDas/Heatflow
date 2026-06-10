import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    Standard residual processing block designed for robust gradient propagation 
    and deep feature extraction across physics-informed networks.
    """
    def __init__(self, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.SiLU()  # Preserves stable, continuous second-order derivatives

    def forward(self, x):
        residual = x
        out = self.activation(self.linear1(x))
        out = self.linear2(out)
        return self.activation(out + residual)


class GeothermalPINN(nn.Module):
    """
    Production Geothermal Physics-Informed Neural Network (PINN).
    """
    def __init__(self, continuous_dim, categorical_cardinalities, embedding_dims=None, 
                 hidden_dim=256, num_residual_blocks=3):
        """
        Args:
            continuous_dim (int): Explicit dimensions of incoming continuous features.
            categorical_cardinalities (list of int): Max size for each categorical feature.
        """
        super(GeothermalPINN, self).__init__()
        
        self.continuous_dim = continuous_dim
        
        # 1. Categorical Feature Projection Engine
        if embedding_dims is None:
            embedding_dims = [max(4, min(32, (c + 1) // 2)) for c in categorical_cardinalities]
            
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=cardinality, embedding_dim=dim)
            for cardinality, dim in zip(categorical_cardinalities, embedding_dims)
        ])
        
        # 2. Input Layer Consolidation
        total_categorical_dim = sum(embedding_dims)
        total_input_dim = 3 + self.continuous_dim + total_categorical_dim
        
        # 3. Network Architecture Trunk
        self.input_layer = nn.Linear(total_input_dim, hidden_dim)
        self.activation = nn.SiLU()
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_residual_blocks)
        ])
        
        self.intermediate_dense = nn.Linear(hidden_dim, hidden_dim // 2)
        self.output_layer = nn.Linear(hidden_dim // 2, 1)

    def forward(self, spatial_coords, geo_features, categorical_indices):
        """
        Executes the network's forward pass evaluation.
        """
        # Process and concatenate embedding tensors
        embedded_outputs = []
        for i, embed_layer in enumerate(self.embeddings):
            embedded_outputs.append(embed_layer(categorical_indices[:, i]))
        
        # Isolate exactly the slice width matching the configured continuous dimension
        pure_continuous = geo_features[:, :self.continuous_dim]
        
        # Merge physical spatial context, continuous inputs, and categorical embeddings
        x = torch.cat([spatial_coords, pure_continuous] + embedded_outputs, dim=1)
        
        # Forward pass execution through structural modules
        x = self.activation(self.input_layer(x))
        
        for block in self.res_blocks:
            x = block(x)
            
        x = self.activation(self.intermediate_dense(x))
        temperature_field = self.output_layer(x)
        
        return temperature_field
