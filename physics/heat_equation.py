import torch

class CrustalHeatEngine:
    """
    Physics engine for a 3D steady-state crustal heat equation.
    
    Evaluates automated gradients, models Fourier's Law of conductive heat-flux,
    and computes the physical residual mismatch of the governing PDE:
        k * ∇²T + Q = 0
        
    This class accurately handles spatial coordinate transformations by applying 
    the chain rule using tracking scale factors to map normalized network domains 
    ([-1, 1]) back to real physical spatial systems (km).
    """
    def __init__(self, lat_scale=1.0, lon_scale=1.0, depth_scale=1.0):
        """
        Args:
            lat_scale (float): Physical radius half-width mapping factor for latitude (km).
            lon_scale (float): Physical radius half-width mapping factor for longitude (km).
            depth_scale (float): Physical radius half-width mapping factor for depth (km).
            
            Note: If spatial normalization maps real coordinates from [min, max] to [-1, 1],
                  then the corresponding scale factor is computed as: (max - min) / 2.0
        """
        self.lat_scale = lat_scale
        self.lon_scale = lon_scale
        self.depth_scale = depth_scale

    def compute_fourier_heat_flow(self, temperature, spatial_coords, k_prior):
        """
        Computes downward conductive heat-flux using Fourier's Law: q = -k * dT/dz
        
        Args:
            temperature (torch.Tensor): Predicted scalar field values [N, 1] in °C.
            spatial_coords (torch.Tensor): Normalized input coordinates [N, 3] as [lat, lon, depth].
            k_prior (torch.Tensor): Vector containing structural thermal conductivity [N, 1] in W/(m·K).
            
        Returns:
            torch.Tensor: Derived vertical heat flow values matrix [N, 1] in mW/m².
        """
        # Isolate the first-order partial derivatives (Jacobian row)
        # create_graph=True preserves structural tracking for downstream second-order calculations
        grads = torch.autograd.grad(
            outputs=temperature.sum(),
            inputs=spatial_coords,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Extract the vertical derivative from the third index column [lat=0, lon=1, depth=2]
        dT_dnorm_z = grads[:, 2].unsqueeze(1)
        
        # Apply the chain rule to convert normalized depth steps back to real-world kilometers
        dT_dz_km = dT_dnorm_z / self.depth_scale
        
        # Convert kilometers to standard meters to align with international conductivity scales:
        # (dT/dz_m) = (dT/dz_km) / 1000.0
        dT_dz_m = dT_dz_km / 1000.0
        
        # Calculate Fourier's conductive heat flux:
        # q = -k * (dT/dz_m). Multiply by 1000.0 to convert units from W/m² to standard mW/m².
        heat_flow_mw = -k_prior * dT_dz_m * 1000.0
        
        return heat_flow_mw

    def compute_laplacian_and_residual(self, temperature, spatial_coords, k_prior, q_prior):
        """
        Computes individual second-order spatial derivatives to construct 
        the 3D Laplacian matrix and maps the steady-state crustal PDE residual.
        
        Args:
            temperature (torch.Tensor): Predicted scalar field values [N, 1] in °C.
            spatial_coords (torch.Tensor): Normalized input coordinates [N, 3] as [lat, lon, depth].
            k_prior (torch.Tensor): Vector containing structural thermal conductivity [N, 1] in W/(m·K).
            q_prior (torch.Tensor): Crustal radiogenic source generation profiles [N, 1] in μW/m³.
            
        Returns:
            dict: Dictionary bundling the core physics layers:
                - 'pde_residual': Physical residual structural tensor [N, 1].
                - 'laplacian': Unified 3D second derivative field [N, 1] in °C/m².
                - 'd2T_dz2': Expressed physical vertical curvature [N, 1] in °C/m².
        """
        # Step 1: Extract first-order derivatives matrix
        grads = torch.autograd.grad(
            outputs=temperature.sum(),
            inputs=spatial_coords,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        dT_dnorm_x = grads[:, 0]
        dT_dnorm_y = grads[:, 1]
        dT_dnorm_z = grads[:, 2]

        # Step 2: Extract separate second-order spatial derivatives (Hessian matrix components)
        grad_outputs = torch.ones_like(dT_dnorm_x)
        
        d2T_dnorm_x2 = torch.autograd.grad(
            outputs=dT_dnorm_x,
            inputs=spatial_coords,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0][:, 0].unsqueeze(1)
        
        d2T_dnorm_y2 = torch.autograd.grad(
            outputs=dT_dnorm_y,
            inputs=spatial_coords,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0][:, 1].unsqueeze(1)
        
        d2T_dnorm_z2 = torch.autograd.grad(
            outputs=dT_dnorm_z,
            inputs=spatial_coords,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0][:, 2].unsqueeze(1)
               
        # Step 3: Apply structural scaling factors to convert coordinates back to physical values
        # Division is squared here to account for second-order spatial steps (d²x, d²y, d²z)
        d2T_dx2_km = d2T_dnorm_x2 / (self.lat_scale ** 2)
        d2T_dy2_km = d2T_dnorm_y2 / (self.lon_scale ** 2)
        d2T_dz2_km = d2T_dnorm_z2 / (self.depth_scale ** 2)
        
        # Convert physical units from kilometers back to standard meters (1 km² = 1,000,000 m²)
        d2T_dx2_m = d2T_dx2_km / 1e6
        d2T_dy2_m = d2T_dy2_km / 1e6
        d2T_dz2_m = d2T_dz2_km / 1e6
        
        # Step 4: Construct the unified 3D crustal Laplacian: ∇²T = d²T/dx² + d²T/dy² + d²T/dz²
        laplacian_m = d2T_dx2_m + d2T_dy2_m + d2T_dz2_m
        
        # Step 5: Evaluate the governing steady-state equation mismatch:
        # k * ∇²T + Q = 0
        # q_prior is converted from μW/m³ to W/m³ (multiplied by 1e-6) to match the thermal conductivity units.
        pde_residual = (k_prior * laplacian_m * 1e6) + q_prior
        
        return {
            'pde_residual': pde_residual,
            'laplacian': laplacian_m,
            'd2T_dz2': d2T_dz2_m
        }
