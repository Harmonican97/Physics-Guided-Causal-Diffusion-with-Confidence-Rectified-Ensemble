import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block1D(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, num_fault_components):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.transform = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.label_mlp = nn.Linear(time_emb_dim, out_ch) 
        
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t, class_emb):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu(h)
        
        time_emb = self.relu(self.time_mlp(t))
        label_emb = self.relu(self.label_mlp(class_emb))
        
        h = h + time_emb.unsqueeze(-1) + label_emb.unsqueeze(-1)
        
        h = self.transform(h)
        h = self.bn2(h)
        h = self.relu(h)
        return h

# ==========================================
# CausalUNet1D
# ==========================================
class CausalUNet1D(nn.Module):
    def __init__(self, input_dim=1, base_channels=64, channel_mults=(1, 2, 4, 8), num_fault_components=3):
        super().__init__()
        # num_fault_components = 3 (Ball, Inner, Outer)
        
        time_emb_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        self.class_encoder = nn.Sequential(
            nn.Linear(num_fault_components, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        self.init_conv = nn.Conv1d(input_dim, base_channels, 3, padding=1)

        self.downs = nn.ModuleList([])
        in_ch = base_channels
        for mult in channel_mults:
            out_ch = base_channels * mult
            self.downs.append(nn.ModuleList([
                Block1D(in_ch, out_ch, time_emb_dim, num_fault_components),
                nn.Conv1d(out_ch, out_ch, 4, 2, 1)
            ]))
            in_ch = out_ch

        self.mid_block1 = Block1D(in_ch, in_ch, time_emb_dim, num_fault_components)
        self.mid_block2 = Block1D(in_ch, in_ch, time_emb_dim, num_fault_components)

        self.ups = nn.ModuleList([])
        for mult in reversed(channel_mults):
            out_ch = base_channels * mult
            self.ups.append(nn.ModuleList([
                nn.ConvTranspose1d(in_ch, out_ch, 4, 2, 1),
                Block1D(out_ch * 2, out_ch, time_emb_dim, num_fault_components)
            ]))
            in_ch = out_ch

        self.out_conv = nn.Conv1d(base_channels, 1, 3, padding=1)

    def forward(self, x, t, y_multihot):
        # y_multihot: (B, 3) float tensor
        t_emb = self.time_mlp(t)
        c_emb = self.class_encoder(y_multihot)

        x = self.init_conv(x)
        skips = []
        for block, downsample in self.downs:
            x = block(x, t_emb, c_emb)
            skips.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t_emb, c_emb)
        x = self.mid_block2(x, t_emb, c_emb)

        for upsample, block in self.ups:
            x = upsample(x)
            skip = skips.pop()
            if x.shape[-1] != skip.shape[-1]:
                x = F.pad(x, (0, 1))
            x = torch.cat((x, skip), dim=1)
            x = block(x, t_emb, c_emb)

        return self.out_conv(x)

class PhysicsLoss:
    @staticmethod
    def envelope_spectrum_loss(signal, target_freqs_list, fs=20480, harmonics=3):
        """
        harmonics
        """
        B, C, L = signal.shape
        envelope = torch.abs(signal)
        fft_res = torch.fft.rfft(envelope, dim=-1, norm='ortho')
        fft_mag = torch.abs(fft_res)
        freq_axis = torch.fft.rfftfreq(L, d=1/fs).to(signal.device)
        
        loss = 0
        valid_samples = 0
        
        for i in range(B):
            freqs = target_freqs_list[i]
            freqs = [f for f in freqs if f > 1.0]
            
            if not freqs:
                continue 
            
            sample_loss = 0
            total_energy = torch.sum(fft_mag[i, 0, 5:]) + 1e-8
            
            for f_base in freqs:
                for k in range(1, harmonics + 1):
                    f_target = f_base * k
                    
                    if f_target > fs / 2:
                        continue

                    idx = (torch.abs(freq_axis - f_target)).argmin()
                    
                    start = max(0, idx - 6)
                    end = min(fft_mag.shape[-1], idx + 7)
                    
                    target_energy = torch.sum(fft_mag[i, 0, start:end])
                    
                    weight = 1.0 / (k ** 0.5)
                    sample_loss += -weight * torch.log(target_energy / total_energy + 1e-10)
            
            loss += sample_loss / (len(freqs) * harmonics)
            valid_samples += 1
            
        if valid_samples == 0:
            return torch.tensor(0.0, device=signal.device, requires_grad=True)
            
        return loss / valid_samples

class PhysicsGuidedDiffusion(nn.Module):
    def __init__(self, model, timesteps=1000, device='cuda'):
        super().__init__()
        self.model = model.to(device)
        self.device = device
        self.timesteps = timesteps
        self.betas = torch.linspace(0.0001, 0.02, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def forward_loss(self, x0, t, y_multihot):
        noise = torch.randn_like(x0)
        x_noised = (self.sqrt_alphas_cumprod[t][:, None, None] * x0 + 
                    self.sqrt_one_minus_alphas_cumprod[t][:, None, None] * noise)
        noise_pred = self.model(x_noised, t, y_multihot)
        return F.mse_loss(noise_pred, noise)

    def physics_guided_sample(self, shape, target_multihot, target_freqs_list, guidance_scale=10.0):
        """
        target_multihot: (B, 3) -> [1,1,1] for Mix
        target_freqs_list: list of [f1, f2, f3] for Mix
        """
        batch_size = shape[0]
        img = torch.randn(shape, device=self.device)
        
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            
            # --- Gradient Calculation ---
            img = img.detach().requires_grad_(True)
            predicted_noise = self.model(img, t, target_multihot)
            
            # Physics Guidance
            alpha_bar = self.alphas_cumprod[t][:, None, None]
            pred_x0 = (img - torch.sqrt(1 - alpha_bar) * predicted_noise) / torch.sqrt(alpha_bar)
            
            phy_loss = PhysicsLoss.envelope_spectrum_loss(pred_x0, target_freqs_list)
            
            grad = torch.autograd.grad(phy_loss, img)[0]
            
            modified_noise = predicted_noise - guidance_scale * torch.sqrt(1 - alpha_bar) * grad
            
            # --- Update ---
            img = img.detach()
            beta = self.betas[i]
            alpha = self.alphas[i]
            
            if i > 0:
                noise = torch.randn_like(img)
            else:
                noise = 0
            
            # DDPM Update
            mean = (1 / torch.sqrt(alpha)) * (img - (beta / torch.sqrt(1 - alpha_bar)) * modified_noise)
            img = mean + torch.sqrt(beta) * noise
            
        return img
    
    def physics_guided_sample_from_t(self, x_t, t_start, target_multihot, target_freqs_list, guidance_scale=10.0):
        """
        guidance_scale
        """
        img = x_t
        batch_size = img.shape[0]
        
        if isinstance(guidance_scale, float):
            scale_tensor = torch.full((batch_size, 1, 1), guidance_scale, device=self.device)
        else:
            scale_tensor = guidance_scale.view(batch_size, 1, 1) # Ensure (B, 1, 1)
        
        for i in reversed(range(0, t_start)):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            
            # --- Gradient Calculation ---
            img = img.detach().requires_grad_(True)
            predicted_noise = self.model(img, t, target_multihot)
            
            alpha_bar = self.alphas_cumprod[t][:, None, None]
            pred_x0 = (img - torch.sqrt(1 - alpha_bar) * predicted_noise) / torch.sqrt(alpha_bar)
            
            # Physics Loss
            phy_loss = PhysicsLoss.envelope_spectrum_loss(pred_x0, target_freqs_list)
            grad = torch.autograd.grad(phy_loss, img)[0]
            
            modified_noise = predicted_noise - scale_tensor * torch.sqrt(1 - alpha_bar) * grad
            
            # --- Update ---
            img = img.detach()
            beta = self.betas[i]
            alpha = self.alphas[i]
            
            if i > 0:
                noise = torch.randn_like(img)
            else:
                noise = 0
            
            mean = (1 / torch.sqrt(alpha)) * (img - (beta / torch.sqrt(1 - alpha_bar)) * modified_noise)
            img = mean + torch.sqrt(beta) * noise
            
        return img
    
    def sample(self, shape, target_multihot):
        batch_size = shape[0]
        img = torch.randn(shape, device=self.device)
        
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            
            predicted_noise = self.model(img, t, target_multihot)
            
            beta = self.betas[i]
            alpha = self.alphas[i]
            alpha_bar = self.alphas_cumprod[t][:, None, None]
            
            mean = (1 / torch.sqrt(alpha)) * (img - (beta / torch.sqrt(1 - alpha_bar)) * predicted_noise)
            
            if i > 0:
                noise = torch.randn_like(img)
            else:
                noise = 0
            
            img = mean + torch.sqrt(beta) * noise
            
        return img