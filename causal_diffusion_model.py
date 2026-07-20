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
# CompositionalUNet1D
# ==========================================
class CompositionalUNet1D(nn.Module):
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
    def analytic_envelope(signal):
        """Return a differentiable Hilbert envelope along the last axis."""
        length = signal.shape[-1]
        centered = signal - signal.mean(dim=-1, keepdim=True)
        spectrum = torch.fft.fft(centered, dim=-1)
        hilbert_filter = torch.zeros(length, device=signal.device, dtype=signal.dtype)
        hilbert_filter[0] = 1.0
        if length % 2 == 0:
            hilbert_filter[length // 2] = 1.0
            hilbert_filter[1:length // 2] = 2.0
        else:
            hilbert_filter[1:(length + 1) // 2] = 2.0
        analytic = torch.fft.ifft(spectrum * hilbert_filter, dim=-1)
        return torch.abs(analytic)

    @staticmethod
    def envelope_spectrum_loss(
        signal,
        target_freqs_list,
        target_weights_list=None,
        fs=20480.0,
        harmonics=3,
        bandwidth_hz=None,
        min_frequency_hz=5.0,
        eps=1e-8,
        band_definition=None,
    ):
        """Transfer-path-weighted characteristic-band envelope loss.

        The input is demeaned, converted to its analytic-signal envelope,
        envelope-demeaned, Hann-windowed, and transformed to a one-sided
        power spectrum. ``target_weights_list`` contains fixed component
        weights determined before training; weights are normalized within
        each requested compound-fault composition.
        """
        B, C, L = signal.shape
        envelope = PhysicsLoss.analytic_envelope(signal)
        envelope = envelope - envelope.mean(dim=-1, keepdim=True)
        if band_definition is None:
            band_definition = PhysicsLoss.build_band_definition(
                target_freqs_list,
                target_weights_list,
                length=L,
                device=signal.device,
                dtype=signal.dtype,
                fs=fs,
                harmonics=harmonics,
                bandwidth_hz=bandwidth_hz,
                min_frequency_hz=min_frequency_hz,
                eps=eps,
            )
        window = band_definition['window']
        envelope = envelope * window.view(1, 1, -1)
        fft_res = torch.fft.rfft(envelope, dim=-1, norm='ortho')
        fft_power = torch.abs(fft_res[:, 0]).pow(2)
        global_energy = fft_power[:, band_definition['global_mask']].sum(dim=1).clamp_min(eps)
        band_energy = (
            fft_power[:, None, :] * band_definition['band_masks'].to(fft_power.dtype)
        ).sum(dim=-1)
        ratios = (band_energy + eps) / (global_energy[:, None] + eps)
        weights = band_definition['band_weights']
        weight_sums = weights.sum(dim=1)
        valid_samples = weight_sums > 0
        if not torch.any(valid_samples):
            return torch.tensor(0.0, device=signal.device, requires_grad=True)

        sample_losses = -(weights * torch.log(ratios)).sum(dim=1) / weight_sums.clamp_min(eps)
        return sample_losses[valid_samples].mean()

    @staticmethod
    def build_band_definition(
        target_freqs_list,
        target_weights_list,
        length,
        device,
        dtype,
        fs,
        harmonics,
        bandwidth_hz,
        min_frequency_hz,
        eps,
    ):
        """Precompute fixed masks and weights for batched physics guidance."""
        if bandwidth_hz is None:
            bandwidth_hz = fs / length
        freq_axis = torch.fft.rfftfreq(length, d=1 / fs).to(device=device)
        rows = []
        for sample_index, raw_freqs in enumerate(target_freqs_list):
            valid_positions = [j for j, value in enumerate(raw_freqs) if value > 1.0]
            if target_weights_list is None:
                component_weights = torch.ones(len(valid_positions), device=device, dtype=dtype)
            else:
                raw_weights = target_weights_list[sample_index]
                component_weights = torch.tensor(
                    [float(raw_weights[j]) for j in valid_positions], device=device, dtype=dtype
                )
            component_weights = component_weights / component_weights.sum().clamp_min(eps)
            bands = []
            for component_index, position in enumerate(valid_positions):
                base_frequency = float(raw_freqs[position])
                for harmonic in range(1, harmonics + 1):
                    target_frequency = base_frequency * harmonic
                    if target_frequency > fs / 2:
                        continue
                    mask = torch.abs(freq_axis - target_frequency) <= bandwidth_hz
                    if not torch.any(mask):
                        nearest = torch.abs(freq_axis - target_frequency).argmin()
                        mask = torch.zeros_like(freq_axis, dtype=torch.bool)
                        mask[nearest] = True
                    weight = component_weights[component_index] / math.sqrt(harmonic)
                    bands.append((mask, weight))
            rows.append(bands)

        max_bands = max((len(row) for row in rows), default=0)
        band_masks = torch.zeros(
            len(rows), max_bands, len(freq_axis), device=device, dtype=torch.bool
        )
        band_weights = torch.zeros(len(rows), max_bands, device=device, dtype=dtype)
        for sample_index, row in enumerate(rows):
            for band_index, (mask, weight) in enumerate(row):
                band_masks[sample_index, band_index] = mask
                band_weights[sample_index, band_index] = weight
        return {
            'window': torch.hann_window(length, periodic=True, device=device, dtype=dtype),
            'global_mask': freq_axis >= min_frequency_hz,
            'band_masks': band_masks,
            'band_weights': band_weights,
        }

class PhysicsGuidedDiffusion(nn.Module):
    def __init__(
        self,
        model,
        timesteps=1000,
        device='cuda',
        fs=20480.0,
        physics_bandwidth_hz=None,
        physics_harmonics=3,
    ):
        super().__init__()
        self.model = model.to(device)
        self.device = device
        self.timesteps = timesteps
        self.fs = float(fs)
        self.physics_bandwidth_hz = physics_bandwidth_hz
        self.physics_harmonics = int(physics_harmonics)
        self._physics_band_cache = {}
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

    def _physics_loss(self, signal, target_freqs_list, target_weights_list):
        frequency_key = tuple(
            tuple(float(value) for value in sample) for sample in target_freqs_list
        )
        weight_key = None if target_weights_list is None else tuple(
            tuple(float(value) for value in sample) for sample in target_weights_list
        )
        cache_key = (
            signal.shape[-1], str(signal.device), str(signal.dtype), frequency_key, weight_key
        )
        if cache_key not in self._physics_band_cache:
            self._physics_band_cache[cache_key] = PhysicsLoss.build_band_definition(
                target_freqs_list,
                target_weights_list,
                length=signal.shape[-1],
                device=signal.device,
                dtype=signal.dtype,
                fs=self.fs,
                harmonics=self.physics_harmonics,
                bandwidth_hz=self.physics_bandwidth_hz,
                min_frequency_hz=5.0,
                eps=1e-8,
            )
        return PhysicsLoss.envelope_spectrum_loss(
            signal,
            target_freqs_list,
            target_weights_list=target_weights_list,
            fs=self.fs,
            harmonics=self.physics_harmonics,
            bandwidth_hz=self.physics_bandwidth_hz,
            band_definition=self._physics_band_cache[cache_key],
        )

    def physics_guided_sample(
        self,
        shape,
        target_multihot,
        target_freqs_list,
        target_weights_list=None,
        guidance_scale=10.0,
    ):
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
            
            phy_loss = self._physics_loss(pred_x0, target_freqs_list, target_weights_list)
            
            grad = torch.autograd.grad(phy_loss, img)[0]
            
            # Adding the loss gradient to epsilon makes the subsequent DDPM
            # mean update move in the *negative* loss-gradient direction.
            # The previous minus sign performed gradient ascent on the
            # envelope loss and therefore suppressed the requested bands.
            modified_noise = predicted_noise + guidance_scale * torch.sqrt(1 - alpha_bar) * grad
            
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
    
    def physics_guided_sample_from_t(
        self,
        x_t,
        t_start,
        target_multihot,
        target_freqs_list,
        target_weights_list=None,
        guidance_scale=10.0,
    ):
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
            phy_loss = self._physics_loss(pred_x0, target_freqs_list, target_weights_list)
            grad = torch.autograd.grad(phy_loss, img)[0]
            
            # See physics_guided_sample: the plus sign here yields gradient
            # descent on the physical loss after substitution into the DDPM
            # reverse mean.
            modified_noise = predicted_noise + scale_tensor * torch.sqrt(1 - alpha_bar) * grad
            
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

    @torch.no_grad()
    def sample_from_t(self, x_t, t_start, target_multihot):
        """Reverse a partially noised signal without physics guidance.

        This is the corrected-protocol sampler used by the compositional-DDPM
        ablation.  Keeping it separate from ``physics_guided_sample_from_t``
        avoids computing a physical-loss gradient and then multiplying it by
        zero when the guidance scale is ``s=0``.
        """
        img = x_t
        batch_size = img.shape[0]
        for i in reversed(range(0, t_start)):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            predicted_noise = self.model(img, t, target_multihot)
            beta = self.betas[i]
            alpha = self.alphas[i]
            alpha_bar = self.alphas_cumprod[t][:, None, None]
            mean = (1 / torch.sqrt(alpha)) * (
                img - (beta / torch.sqrt(1 - alpha_bar)) * predicted_noise
            )
            noise = torch.randn_like(img) if i > 0 else 0
            img = mean + torch.sqrt(beta) * noise
        return img


# Backward-compatible import name for existing scripts and checkpoints.
CausalUNet1D = CompositionalUNet1D
