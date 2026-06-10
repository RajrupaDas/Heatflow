import torch
import torch.nn as nn
from typing import List, Dict, Tuple

class ResidualBlock(nn.Module):
    """
    Standard pre-activation Residual Block utilizing SiLU activation.
    Ensures stable gradient flow for high-order autograd physics terms.
    """
    def __init__(self, hidden_size: int = 256, dropout_rate: float = 0.05):
        super().__init__()
        self.layer1 = nn.Linear(hidden_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # Layer normalization for internal training stability
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        # First layer
        out = self.ln1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.layer1(out)
        
        # Second layer
        out = self.ln2(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.layer2(out)
        
        # Skip connection
        return out + residual


class GeothermalPINN(nn.Module):
    """
    Research-grade PINN Model architecture for predicting Surface Heat Flow directly across India.
    Integrates categorical embeddings with a deep residual feature backbone.
    """
    def __init__(
        self,
        num_numerical_features: int = 11,
        cat_cardinalities: List[int] = [4, 5, 4],
        embedding_dims: List[int] = [16, 16, 8],   # kept for API compat, unused
        hidden_size: int = 256,
        depth_blocks: int = 4,
        dropout_rate: float = 0.05,
        total_input_dim: int = None  # pass this from train.py
    ):
        super().__init__()

        self.num_numerical_features = num_numerical_features
        self.cat_cardinalities = cat_cardinalities

        # The preprocessor outputs one-hot encoded cats, not indices.
        # total_input_dim must be passed in (num_numerical + num_onehot_cols).
        if total_input_dim is None:
            raise ValueError("total_input_dim must be provided (num_numerical_features + total one-hot columns).")

        # 2. Input Projection Layer
        self.input_projection = nn.Linear(total_input_dim, hidden_size)
           
        # 3. Residual MLP Backbone
        self.backbone = nn.Sequential(*[
            ResidualBlock(hidden_size=hidden_size, dropout_rate=dropout_rate)
            for _ in range(depth_blocks)
        ])
        
        # 4. Output Head (Predicts Surface Heat Flow directly; Scalar)
        self.final_ln = nn.LayerNorm(hidden_size)
        self.final_activation = nn.SiLU()
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x_numerical: torch.Tensor, x_categorical: torch.Tensor) -> torch.Tensor:
        """
        Forward pass processing spatial/geological parameters to map directly to Surface Heat Flow.
        
        Parameters:
        -----------
        x_numerical : torch.Tensor
            Tensor of shape (B, num_numerical_features), including lat, lon, thicknesses, etc.
        x_categorical : torch.Tensor
            Tensor of shape (B, len(cat_cardinalities)) containing label-encoded indices for embedding lookups.
            
        Returns:
        --------
        predicted_heat_flow : torch.Tensor
            Scalar heat-flow predictions of shape (B, 1) in mW/m².
        """
        # Concatenate numerical fields with embedding vectors
        combined_features = torch.cat([x_numerical, x_categorical], dim=1)

        # Project to hidden space
        h = self.input_projection(combined_features)
        
        # Pass through Deep Residual Backbone
        h = self.backbone(h)
        
        # Final pre-output conditioning
        h = self.final_ln(h)
        h = self.final_activation(h)
        
        # Direct generation of target heat flow value
        predicted_heat_flow = self.output_layer(h)
        return predicted_heat_flow


# Verification and Spatial Autograd Integration Test
if __name__ == "__main__":
    print("--- Verifying Model Architectural Pipeline ---")
    
    # Configuration matches your specified inputs
    num_num_features = 11  # lat, lon, crust_thickness, sediment_thickness, etc.
    mock_cardinalities = [5, 8, 4]  # geo_lithology, geo_stratigraphy, slip_type categories
    mock_emb_dims = [16, 16, 8]
    
    # Initialize PINN Network
    model = GeothermalPINN(
        num_numerical_features=num_num_features,
        cat_cardinalities=mock_cardinalities,
        embedding_dims=mock_emb_dims,
        hidden_size=256,
        depth_blocks=4
    )
    
    print(model)
    
    # Generate mock tensor batch representing raw dataset structures
    batch_size = 32
    
    # Mock inputs (Normal distributed values)
    mock_x_num = torch.randn(batch_size, num_num_features)
    # Ensure spatial coordinates explicitly track graph histories for PDE evaluation
    mock_x_num.requires_grad_(True) 
    
    # Categorical allocations bounded within structural cardinality ceilings
    mock_x_cat = torch.stack([
        torch.randint(0, mock_cardinalities[0], (batch_size,)),
        torch.randint(0, mock_cardinalities[1], (batch_size,)),
        torch.randint(0, mock_cardinalities[2], (batch_size,))
    ], dim=1)

    # 1. Model Prediction Test
    q_pred = model(mock_x_num, mock_x_cat)
    print(f"\nForward verification pass shape: {q_pred.shape} (Expected: [{batch_size}, 1])")
    
    # 2. Physics Term Validation: Compute Laplacian (∇²q) via Autograd
    print("\n--- Verifying Autograd Gradient Graph for Physics Term ---")
    
    # Gradient of heat flow with respect to input features
    grad_q = torch.autograd.grad(
        outputs=q_pred,
        inputs=mock_x_num,
        grad_outputs=torch.ones_like(q_pred),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Spatial feature slices matching preprocessing design (Lat=Index 0, Lon=Index 1)
    dq_dlat = grad_q[:, 0:1]
    dq_dlon = grad_q[:, 1:2]
    
    # Compute Second Derivatives (Laplacian terms)
    d2q_dlat2 = torch.autograd.grad(
        outputs=dq_dlat,
        inputs=mock_x_num,
        grad_outputs=torch.ones_like(dq_dlat),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0][:, 0:1]
    
    d2q_dlon2 = torch.autograd.grad(
        outputs=dq_dlon,
        inputs=mock_x_num,
        grad_outputs=torch.ones_like(dq_dlon),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0][:, 1:2]
    
    laplacian_q = d2q_dlat2 + d2q_dlon2
    
    # Compute Physics Loss Term: mean((∇²q)^2)
    physics_loss = torch.mean(laplacian_q ** 2)
    print(f"Calculated Physics Loss Scalar: {physics_loss.item():.6f}")
    
    # Backward verification to ensure optimization compatibility
    physics_loss.backward()
    print("Backward graph pass through structural residuals completed successfully.")
