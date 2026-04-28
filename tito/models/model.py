from tqdm import tqdm
import torch
import torch_geometric as geom
import lightning as pl

from tito.models import utils
from tito.data.datasets import BaseDensity


class CFM(pl.pytorch.LightningModule):
    def __init__(self, score, lr=1e-3):#, basedistribution, ot_coupling=True, ot_plan="rp"):
        super().__init__()
        #self.save_hyperparameters(ignore=["basedistribution"])
        self.score = score #TODO: change score to vf
        self.sigma = 0.001
        self.save_hyperparameters()
        self.learning_rate = lr
        #self.__basedistribution = basedistribution
        #self.ot_coupling = ot_coupling
        #self.ot_plan = ot_plan


    def training_step(self, batch, batch_idx):
        t = torch.rand(len(batch['cond'])).type_as(batch['cond'].x)
        loss = self.get_loss(t, batch)
        self.log("train/loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        t = torch.rand(len(batch['cond'])).type_as(batch['cond'].x)
        loss = self.get_loss(t, batch)
        self.log("val/loss", loss, prog_bar=True)
        return loss            

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def _forward(self, t, batch):
        batch["t_diff"] = t
        return self.score(t, batch)

    def get_loss(self, t, batch):
        batch['corr'] = batch['cond'].clone() # clone batch for interpolated coordinates 
        x0 = batch['target'].xbase #self.__basedistribution.sample_as(batch['cond']) # sample from base distribution
        x1 = batch['target'].x # set target interpolation coordinates 
        t_batch = t[batch['cond'].batch] # associate coordinates with sampled times

        xt = self.sample_conditional_pt(t_batch, x0, x1, batch=batch['cond'].batch) # compute interpolated coordinates
 
        batch['corr'].x = xt # inject interpolated coordinates into batch
        ut = self.compute_conditional_vector_field(x0, x1) # compute vector field

        vt = self._forward(t, batch) # predict vector field from model

        norms = torch.norm(vt - ut, dim=1) # compute norm of difference between predicted and conditional vector field
        loss = torch.mean(norms**2) # mse loss
        return loss

    def sample_conditional_pt(self, t, x0, x1, batch):
        epsilon = torch.normal(0, 1, size=x0.shape, device=x0.device)
        epsilon = utils.center_coordinates_batch(epsilon, batch) 
        mu_t = (t * x1.T + (1 - t) * x0.T).T
        return mu_t + self.sigma * epsilon

    def compute_conditional_vector_field(self, x0, x1):
        return x1 - x0

    def sample(
        self,
        batch,
        ode_steps=50,
        nested_samples=1,
        base_distribution=BaseDensity(std=1.0),
        ode_solver="euler",
        center_each_step=True,
        max_step_displacement=None,
    ):
        self.eval()
        with torch.no_grad():
            if ode_steps <= 0:
                raise ValueError("ode_steps must be positive.")
            ode_solver = ode_solver.lower()
            if ode_solver not in {"euler", "heun"}:
                raise ValueError("ode_solver must be 'euler' or 'heun'.")

            device = next(self.parameters()).device
            sh = SampleHandler(self._forward)
            self.score.eval()
            #self.score.training = False
            
            x0 = batch['corr'].x.clone()
            dt = 1.0 / ode_steps
            traj = [batch['cond'].x.clone()]

            for i_nested in tqdm(range(nested_samples)):
                #print(f'Sampling nested step {i_nested+1}/{nested_samples}...')
                for i_ode in range(ode_steps): # simple Forward Euler solver
                    #print(f'Sampling step {i_ode}...', end='\r')
                    t = torch.tensor([i_ode], dtype=x0.dtype, device=device) * dt
                    t_next = torch.tensor([i_ode + 1], dtype=x0.dtype, device=device) * dt
                    x0 = self._ode_step(
                        sh,
                        t,
                        t_next,
                        batch,
                        x0,
                        dt,
                        ode_solver=ode_solver,
                        center_each_step=center_each_step,
                        max_step_displacement=max_step_displacement,
                    )
                    batch['corr'].x = x0
                traj.append(x0.clone())
                batch["cond"].x = x0.clone() # update condition with last step
                #base_samples = torch.normal(0, 1, size=batch["cond"].x.shape)
                #base_samples = utils.center_coordinates_batch(base_samples, batch["cond"].batch) 
                base_samples = base_distribution.sample_as(batch["cond"].x)
                batch["corr"].x = base_samples.clone()
                x0 = base_samples.clone() # reset x0 to base distribution for next nested sample

            batch["traj"] = batch["cond"].clone()
            batch["traj"].x = torch.stack(traj, dim=0) # store trajectory
            print("Done!")
            return batch

    def _ode_step(
        self,
        sample_handler,
        t,
        t_next,
        batch,
        x,
        dt,
        ode_solver,
        center_each_step,
        max_step_displacement,
    ):
        batch["corr"].x = x
        if ode_solver == "euler":
            delta = dt * sample_handler(t, batch)
        else:
            v0 = sample_handler(t, batch)
            proposal = x + dt * v0
            batch["corr"].x = proposal
            v1 = sample_handler(t_next, batch)
            delta = 0.5 * dt * (v0 + v1)

        delta = self._clip_step(delta, max_step_displacement)
        x_next = x + delta
        if center_each_step:
            x_next = utils.center_coordinates_batch(x_next, batch["corr"].batch)
        return x_next

    @staticmethod
    def _clip_step(delta, max_step_displacement):
        if max_step_displacement is None:
            return delta
        if max_step_displacement <= 0:
            raise ValueError("max_step_displacement must be positive when provided.")

        norm = delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        scale = torch.clamp(max_step_displacement / norm, max=1.0)
        return delta * scale
    
class SampleHandler:
    def __init__(self, sample_forward):
        self.sample_forward = sample_forward

    def __call__(self, t, batch):
        t = t.repeat((len(torch.unique(batch['cond'].batch)), ))
        return self.sample_forward(t, batch)
