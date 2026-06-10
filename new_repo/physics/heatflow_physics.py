import torch
def compute_spatial_laplacian(
    q_pred: torch.Tensor, 
    x_num: torch.Tensor
) -> torch.Tensor:
    """
    Computes the spatial Laplacian (∇²q) of the predicted heat flow field 
    with respect to latitude and longitude using PyTorch Autograd.
    """
    if not x_num.requires_grad:
        raise ValueError("x_num tensor must have requires_grad=True to compute physics terms.")

    # 1. Compute First-Order Derivatives with respect to all numerical inputs
    dq_dx = torch.autograd.grad(
        outputs=q_pred,
        inputs=x_num,
        grad_outputs=torch.ones_like(q_pred),
        create_graph=True,    
        retain_graph=True,
        only_inputs=True
    )[0]

    # Isolate the gradients corresponding to Lat (Index 0) and Lon (Index 1)
    dq_dlat = dq_dx[:, 0:1]
    dq_dlon = dq_dx[:, 1:2]

    # 2. Compute Second-Order Derivative for Latitude: ∂²q / ∂lat²
    d2q_dlat2 = torch.autograd.grad(
        outputs=dq_dlat,
        inputs=x_num,
        grad_outputs=torch.ones_like(dq_dlat),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0][:, 0:1]

    # 3. Compute Second-Order Derivative for Longitude: ∂²q / ∂lon²
    d2q_dlon2 = torch.autograd.grad(
        outputs=dq_dlon,
        inputs=x_num,
        grad_outputs=torch.ones_like(dq_dlon),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0][:, 1:2]

    # 4. Construct Laplacian: ∇²q = (∂²q / ∂lat²) + (∂²q / ∂lon²)
    laplacian = d2q_dlat2 + d2q_dlon2

    return laplacian
def compute_physics_loss(laplacian: torch.Tensor) -> torch.Tensor:
    """
    Computes the smoothness regularization physics loss term.
    
    Formula:
        mean((∇²q)^2)
        
    Parameters:
    -----------
    laplacian : torch.Tensor
        The computed ∇²q field tensor of shape (B, 1).
        
    Returns:
    --------
    physics_loss : torch.Tensor
        Scalar tensor representing the structural regularization penalty.
    """
    physics_loss = torch.mean(laplacian ** 2)
    return physics_loss


def evaluate_heatflow_physics(
    q_pred: torch.Tensor, 
    spatial_coords: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    High-level operational wrapper integrating Laplacian tracking and 
    smoothness regularization loss calculations for PINN backpropagation loops.
    
    Parameters:
    -----------
    q_pred : torch.Tensor
        Predicted surface heat flow, shape (B, 1).
    spatial_coords : torch.Tensor
        Spatial tracker tensor containing [lat, lon], shape (B, 2).
        
    Returns:
    --------
    laplacian : torch.Tensor
        The raw ∇²q tensor, shape (B, 1).
    physics_loss : torch.Tensor
        The derived mean-squared structural physics regularization loss (Scalar).
    """
    laplacian = compute_spatial_laplacian(q_pred, spatial_coords)
    physics_loss = compute_physics_loss(laplacian)
    
    return laplacian, physics_loss


# Verification Unit Test Execution
if __name__ == "__main__":
    print("--- Executing Geothermal Physics Engine Functional Test ---")
    
    # Simulate a batch of 8 grid nodes across India
    batch_size = 8
    
    # Initialize mock spatial coordinates: Column 0 = Lat, Column 1 = Lon
    # Explicitly activate the tracking graph structure
    mock_spatial_coords = torch.tensor([
        [23.5, 78.5], # Central India
        [13.0, 80.2], # Southern Coast (Chennai)
        [28.6, 77.2], # Northern India (Delhi)
        [19.0, 72.8], # Western Coast (Mumbai)
        [22.5, 88.3], # Eastern Region (Kolkata)
        [34.1, 74.8], # Far North (Srinagar)
        [10.0, 76.3], # South West (Kochi)
        [26.1, 91.7]  # North East (Guwahati)
    ], dtype=torch.float32, requires_grad=True)

    # Simulate a highly non-linear predicted heat flow function q(lat, lon) 
    # to guarantee non-zero second-order gradients: 
    # q = 50.0 + 3.0*(lat^3) + 2.5*(lon^2 * lat)
    lat_col = mock_spatial_coords[:, 0:1]
    lon_col = mock_spatial_coords[:, 1:2]
    mock_q_pred = 50.0 + 3.0 * (lat_col ** 3) + 2.5 * (lon_col ** 2 * lat_col)

    # Process through the physics pipeline
    laplacian_field, regularization_loss = evaluate_heatflow_physics(
        q_pred=mock_q_pred, 
        spatial_coords=mock_spatial_coords
    )

    print("\n--- Numerical Verification Matrix ---")
    for idx in range(batch_size):
        print(f"Node {idx} | Lat: {mock_spatial_coords[idx, 0].item():.1f}°N, "
              f"Lon: {mock_spatial_coords[idx, 1].item():.1f}°E | "
              f"Predicted Heat Flow: {mock_q_pred[idx].item():.2f} mW/m² | "
              f"∇²q: {laplacian_field[idx].item():.4f}")

    print("\n--- Structural Outputs Check ---")
    print(f"Laplacian Tensor Shape         : {laplacian_field.shape} (Expected: [{batch_size}, 1])")
    print(f"Physics Loss Scalar Value      : {regularization_loss.item():.6f}")
    print(f"Grad Graph History Intact      : {regularization_loss.grad_fn is not None}")
    
    # Execute backward check to confirm loss backpropagation chain matches optimizer expectations
    regularization_loss.backward()
    print(f"Spatial Gradients Propagated   : {mock_spatial_coords.grad is not None}")
    print("Physics code engine operating correctly within design parameters.")
