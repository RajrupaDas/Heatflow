import os
import json
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

class GeothermalPINNTrainer:
    """
    Production training engine for the Geothermal PINN model architecture.
    Handles mixed-precision training, loss scaling, tracking checkpoints,
    early stopping validation, and TensorBoard logging.
    """
    def __init__(self, model, heat_engine, optimizer, config, device='cpu'):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.heat_engine = heat_engine
        self.optimizer = optimizer
        
        # Configuration parameters
        self.epochs = config.get('epochs', 2000)
        self.early_stopping_patience = config.get('patience', 100)
        self.checkpoint_dir = config.get('checkpoint_dir', 'models/checkpoints')
        self.lambda_obs = config.get('lambda_obs', 1.0)
        self.lambda_bc = config.get('lambda_bc', 0.5)
        self.lambda_pde = config.get('lambda_pde', 0.1)
        self.surface_temp_bc = config.get('surface_temp_bc', 25.0) # °C constant baseline
        
        # Performance enhancers
        self.scaler = torch.amp.GradScaler('cuda' if self.device.type == 'cuda' else 'cpu')
        self.writer = SummaryWriter(log_dir=config.get('log_dir', 'runs/geothermal_pinn'))
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.criterion = nn.MSELoss()
        
        # Tracking states
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.history = {'train_loss': [], 'val_loss': [], 'pde_loss': [], 'obs_loss': [], 'bc_loss': []}

    def compute_losses(self, train_batch, collocation_batch, q_prior_colloc):
        """Computes and aggregates all structural components of the multi-task PINN loss."""
        # 1. Borehole Heat-Flow Observation Loss (z = 0)
        spatial_obs = train_batch['spatial_coords'].requires_grad_(True)
        pred_T_obs = self.model(spatial_obs, train_batch['geo_features'], train_batch['categorical_indices'])
        
        pred_q_obs = self.heat_engine.compute_fourier_heat_flow(pred_T_obs, spatial_obs, train_batch['k_prior'])
        loss_obs = self.criterion(pred_q_obs, train_batch['targets'])

        # 2. Surface Temperature Boundary Condition Enforcement (T = 25°C at z = -1.0 normalized surface)
        # FIX: Clone and modify the depth array values BEFORE activating autograd tracking (.requires_grad_)
        spatial_bc = collocation_batch['spatial_coords'].clone().detach()
        spatial_bc[:, 2] = -1.0  # Safe modification: not tracking gradients yet!
        spatial_bc.requires_grad_(True)  # Now lock it down for derivative evaluation
        
        pred_T_bc = self.model(spatial_bc, collocation_batch['geo_features'], collocation_batch['categorical_indices'])
        target_T_bc = torch.full_like(pred_T_bc, self.surface_temp_bc)
        loss_bc = self.criterion(pred_T_bc, target_T_bc)
        
        # 3. Crustal Steady-State Geothermal PDE Loss
        spatial_colloc = collocation_batch['spatial_coords'].clone().detach().requires_grad_(True)
        pred_T_colloc = self.model(spatial_colloc, collocation_batch['geo_features'], collocation_batch['categorical_indices'])
        
        pde_outputs = self.heat_engine.compute_laplacian_and_residual(
            pred_T_colloc, spatial_colloc, collocation_batch['k_prior'], q_prior_colloc
        )

        pde_residual = pde_outputs['pde_residual']
        loss_pde = self.criterion(pde_residual, torch.zeros_like(pde_residual))
        
        # Aggregate total weighted loss structure
        total_loss = (self.lambda_obs * loss_obs) + (self.lambda_bc * loss_bc) + (self.lambda_pde * loss_pde)
        
        return total_loss, loss_obs, loss_bc, loss_pde

    def train_epoch(self, train_batch, collocation_batch, q_prior_colloc):
        """Executes a single mixed-precision optimization pass."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Automated Mixed Precision context
        device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
        with torch.amp.autocast(device_type=device_type):
            total_loss, loss_obs, loss_bc, loss_pde = self.compute_losses(
                train_batch, collocation_batch, q_prior_colloc
            )
            
        self.scaler.scale(total_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return {
            'total': total_loss.item(),
            'obs': loss_obs.item(),
            'bc': loss_bc.item(),
            'pde': loss_pde.item()
        }
        # Force-release graph memory references at the end of every training cycle
        self.optimizer.zero_grad(set_to_none=True) 

        # Flush the hardware cache memory pools explicitly
        torch.cuda.empty_cache()

    @torch.no_grad()
    def validate(self, val_batch):
        """Evaluates model performance on the validation split using observation metrics."""
        self.model.eval()
        spatial_obs = val_batch['spatial_coords'].clone().detach().requires_grad_(True)
        
        with torch.set_grad_enabled(True): # Enable local autograd tracking for Fourier heat-flow evaluation
            pred_T = self.model(spatial_obs, val_batch['geo_features'], val_batch['categorical_indices'])
            pred_q = self.heat_engine.compute_fourier_heat_flow(pred_T, spatial_obs, val_batch['k_prior'])
            val_loss = self.criterion(pred_q, val_batch['targets'])
            
        return val_loss.item()

    def fit(self, train_batch, val_batch, collocation_sampler, preprocessor, q_prior_colloc_val, config):
        """Orchestrates the entire training timeline."""
        print(f"🚀 Initializing PINN optimization on hardware device context: {self.device}")
        
        # Transfer validation background priors to target hardware
        q_prior_colloc = torch.tensor(q_prior_colloc_val, dtype=torch.float32, device=self.device).unsqueeze(1)

        colloc_batch = collocation_sampler.prepare_collocation_tensors( num_points=config.get('collocation_nodes', 20000), preprocessor=preprocessor) 
        
        for epoch in range(1, self.epochs + 1):                        
            # Execute standard step updates
            losses = self.train_epoch(train_batch, colloc_batch, q_prior_colloc)
            val_loss = self.validate(val_batch)
            
            # Record profiles to metrics engine
            self.history['train_loss'].append(losses['total'])
            self.history['val_loss'].append(val_loss)
            self.history['obs_loss'].append(losses['obs'])
            self.history['bc_loss'].append(losses['bc'])
            self.history['pde_loss'].append(losses['pde'])
            
            # Write metrics directly to TensorBoard logs
            self.writer.add_scalar('Loss/Total_Train', losses['total'], epoch)
            self.writer.add_scalar('Loss/Validation_Obs', val_loss, epoch)
            self.writer.add_scalar('Loss/Component_Observation', losses['obs'], epoch)
            self.writer.add_scalar('Loss/Component_Boundary_Condition', losses['bc'], epoch)
            self.writer.add_scalar('Loss/Component_PDE_Residual', losses['pde'], epoch)
            
            if epoch % 50 == 0 or epoch == 1:
                print(f"Epoch {epoch:04d}/{self.epochs} | Total Loss: {losses['total']:.4f} | "
                      f"Obs: {losses['obs']:.4f} | BC: {losses['bc']:.4f} | PDE: {losses['pde']:.4e} | Val Loss: {val_loss:.4f}")
                
            # Check early stopping conditions and manage checkpoints
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint('best_pinn_model.pt', epoch, val_loss)
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.early_stopping_patience:
                print(f"Early stopping conditions met at epoch sequence index {epoch}. Training stopped.")
                break
                
        self.writer.close()
        self.save_history_log()

    def save_checkpoint(self, filename, epoch, val_loss):
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss
        }, path)

    def save_history_log(self):
        log_path = os.path.join(self.checkpoint_dir, 'training_history.json')
        with open(log_path, 'w') as f:
            json.dump(self.history, f, indent=4)
