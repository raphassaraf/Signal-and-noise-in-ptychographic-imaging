import matplotlib.pyplot as plt
import torch as t
import torch.nn as nn

from PIL import Image
from skimage.draw import disk
from torch.optim import Adam, lr_scheduler, LBFGS
from torchvision import transforms

from cdtools.tools import propagators, interactions


def periodic_padding(img):
    '''
    Add periodic padding around to the image.
    '''
    
    pad = nn.CircularPad2d(img.shape[0])
    
    return pad(img.unsqueeze(0)).squeeze()


def simulate_object(path, resize=None, bc=None):
    '''
    Simulate an object complex-valued wavefront from an image.
    '''
    
    sim_obj = transforms.functional.pil_to_tensor(Image.open(path, mode='r'))
    
    if resize is not None:
        sim_obj = transforms.Resize((resize, resize))(sim_obj).permute(1, 2, 0)

    sim_obj = sim_obj[:,:,0] / 255 + 1j * sim_obj[:,:,1] / 255
    
    if bc == 'periodic': sim_obj = periodic_padding(sim_obj) 

    return sim_obj


def get_obj_guess(size):
    '''
    Initialize the initial object guess with all pixels equal to 1+1j.
    '''
    
    obj_guess = (t.ones((size, size)) + 1j*t.zeros((size, size)))

    return obj_guess


def simulate_probe(m, r_ratio, r_inner=0, propagate=False, distance=20, band_lim_rand=False):
    '''Simulates a probe in a square image, based on focused/defocused Fourier Transform.

    Args:
        m: size of the image's side
        r_ratio: ratio between the size of a side and the radius of the donut
        r_inner: radius of the inner circle
        propagate: bool for including propagation
        distance: distance of propagation
        band_lim_rand: bool for simulating a band-limited random probe

    Returns:
        A probe of shape (m, m)'''

    size = (m, m)
    center = (m//2, m//2)
    r_outer = m/r_ratio
    rr_big, cc_big = disk(center, r_outer, shape=size)
    rr_small, cc_small = disk(center, r_inner, shape=size)

    mask = t.zeros(size)
    mask[rr_big, cc_big] = 1
    mask[rr_small, cc_small] = 0

    if band_lim_rand:
        random_aperture = t.zeros(size)
        random_aperture[rr_big, cc_big] = t.normal(0, 1, size=rr_big.shape)
        random_aperture = t.fft.fftshift(t.fft.ifft2(t.fft.ifftshift(random_aperture)))
        probe = random_aperture * mask
        return t.fft.fftshift(t.fft.ifft2(t.fft.ifftshift(probe)))
    
    else:
        probe = t.fft.fftshift(t.fft.ifft2(t.fft.ifftshift(mask)))
        if propagate:
            propagator = propagators.generate_angular_spectrum_propagator([m, m], [1, 1], 1, distance)
            probe = propagators.near_field(probe, propagator)
        return probe


def simulate_multiprobe_grad(m, r_ratio, r_inner=0, propagate=False,
                             distance=None, weights=[100, 0, 0]):
    '''
    Simulate multiprobe tensor from the weighted directional derivatives of an initial probe.
    '''

    probe = simulate_probe(m, r_ratio, r_inner, propagate, distance)
    grad_x, grad_y = t.gradient(probe)

    # Rescale the gradients to match the initial probe's modulus
    probe_area = probe.shape[0] * probe.shape[1]
    probe_scale = t.abs(probe).sum() / probe_area
    x_scale = t.abs(grad_x).sum() / probe_area
    y_scale = t.abs(grad_y).sum() / probe_area

    grad_x = grad_x * probe_scale / x_scale
    grad_y = grad_y * probe_scale / y_scale

    return t.stack([weights[0]*probe, weights[1]*grad_x, weights[2]*grad_y])


def simulate_multiprobe_defocus(m, r_ratio, r_inner=0, propagate=False, propa_distances=None):
    '''
    Simulate multiprobe tensor from the different focal distances of the same probe.
    '''
    
    multiprobes = [
        simulate_probe(m, r_ratio, r_inner, propagate, d)
        for d in propa_distances
    ]

    return t.stack(multiprobes)


def simulate_multiprobe_band_limited_random(m, r_ratio, r_inner=0, n_probes=10):
    '''Simulates a multiprobe tensor with band-limited random probes.
    
    Args:
        m: size of the image's side
        r_ratio: ratio between the size of a side and the radius of the donut
        r_inner: radius of the inner circle
        n_probes: number of simulated modes
    
    Returns:
        A tensor of shape (n_probes, m, m) containing each simulated probe.'''
    
    multiprobes = [
        simulate_probe(m, r_ratio, r_inner, band_lim_rand=True)
        for _ in range(n_probes)
    ]

    return t.stack(multiprobes)   


def set_scanning_grid(coord, n_steps, steps_size, add_noise=True, noise_ratio=8):
    '''
    Initialize a scanning grid with gaussian noise.
    '''

    # Create a perfect scan grid.
    x = t.arange(coord[0], coord[0] + n_steps * steps_size, steps_size)
    y = t.arange(coord[1], coord[1] + n_steps * steps_size, steps_size)
    scan_grid = t.stack(t.meshgrid(x, y), dim=-1).reshape(-1, 2)
    
    # Generate Gaussian noise.
    if add_noise:
        noise = t.normal(
            mean=0,
            std=size / noise_ratio,
            size=(steps**2, 2)
        ).round().int()
    else: noise = 0
    
    # Add noise to the scan grid.
    scan_grid += noise
    
    return scan_grid


def get_intensity(wave):
    '''Calculate the intensity of a wave.
    
    Args:
        wave: 2-dimensional tensor of a wave in real space
        
    Returns: Intensity of the wave's Fourier Transform'''
    
    return t.abs(t.fft.fftshift(t.fft.fft2(t.fft.ifftshift(wave))))**2


def get_diffractions_direct(probe, obj, translations):
    '''
    Calculate diffraction pattern's intensity using ptycho_2D_round.
    '''

    if probe.dim() == 3: # multi-probe intensity
        multiprobe_intensities = t.stack([
            get_intensity(interactions.ptycho_2D_round(p, obj, translations))
            for p in probe
        ])
        diff_patterns = t.sum(multiprobe_intensities, 0)
        
    else:
        diff_patterns = get_intensity(
            interactions.ptycho_2D_round(probe, obj, translations)
        )

    return diff_patterns


def get_diffractions_fluence(probe, obj, translations, fluence, noise=None):
    '''
    Calculate the diffraction patterns for a given probe, object,
    translations grid and fluence.
    '''

    diff_patterns = get_diffractions_direct(probe, obj, translations)

    output_probe = probe
    
    # Scale intensity to match photon fluence
    fluence_i = diff_patterns.sum() / (
        obj.shape[0] * obj.shape[1]
    )
    probe_norm = t.sqrt(fluence / fluence_i)
    output_probe = probe * probe_norm
    diff_patterns = diff_patterns * fluence / fluence_i

    # Add Poisson noise to intensities
    if noise == 'poisson':
        diff_patterns = t.poisson(diff_patterns)
    
    return diff_patterns, output_probe

def mse_amplitude_loss(sim_intensity, measured_intensity):
    '''
    Corrected MSE loss
    '''
    eps = 1e-10 # term to prevent derivative from blowing up during optimization
    loss = t.sum(
        (t.sqrt(sim_intensity+eps) - t.sqrt(measured_intensity+eps))**2
        ) / t.numel(sim_intensity)
    
    return loss
    
    
def AD_model_LBFGS(diffractions, sim_probe, obj_guess, translations, epochs,
             lr=1, patience=5, tolerance=1e-8, loss_f='PoissonNLL', show=False):
    '''
    Reconstruct the object using autodiff maximum likelihood.
    loss_f: {'PoissonNLL', 'MSE'}
    '''

    if loss_f == 'PoissonNLL': loss_fct = nn.PoissonNLLLoss(log_input=False)
    elif loss_f == 'MSE': loss_fct = mse_amplitude_loss
        
    obj_guess.requires_grad = True
    optimizer = LBFGS([obj_guess], lr=lr)
    
    loss_list = []
    best_loss = float('inf')
    patience_counter = 0
    
    def closure_LBFGS():
        
        optimizer.zero_grad()
        
        predicted_patterns = get_diffractions_direct( # correct, but why ? --> annotate
            probe=sim_probe,
            obj=periodic_padding(obj_guess),
            translations=translations,
        )
        
        loss = loss_fct(predicted_patterns, diffractions)
        loss.backward()     
        
        return loss

    for e in range(epochs):
        
        loss = optimizer.step(closure_LBFGS)
        loss_list.append(loss.detach().to('cpu'))
        print(f'Epoch {e}: loss = {loss}')
        
        # Early stopping check
        if loss_list[-1] < best_loss - tolerance:
            best_loss = loss_list[-1]
            patience_counter = 0  # Reset counter if there's an improvement
        else:
            patience_counter += 1
        
        # If no improvement within 'patience' epochs, stop early
        if patience_counter >= patience:
            print(f"Early stopping at epoch {e} with loss {best_loss}")
            break

    return obj_guess.detach().clone().to('cpu'), loss_list