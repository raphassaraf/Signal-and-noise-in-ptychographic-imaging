import matplotlib.pyplot as plt
import torch as t
import torch.nn as nn

from PIL import Image
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


def simulate_probe(m, r_ratio, r_inner=0, propagate=False, distance=None,
                   pixel_size=[1, 1], wavelength=1):
    '''
    Simulate an Airy function probe wavefront.
    '''
    
    r = m/r_ratio
    sim_probe_FT = t.zeros((m,m))
    for x in range(m):
        for y in range(m):
            if ((x-m/2)**2 + (y-m/2)**2 < r**2 and
                    (x-m/2)**2 + (y-m/2)**2 >= r_inner**2):
                sim_probe_FT[x, y] = 1
    probe = t.fft.fftshift(t.fft.ifft2(t.fft.ifftshift(sim_probe_FT)))

    if propagate == True:
        propagator = propagators.generate_angular_spectrum_propagator([m, m], pixel_size, wavelength, distance)
        probe = propagators.near_field(probe, propagator)
                
    return probe


def simulate_multiprobe_grad(m, r_ratio, r_inner=0, propagate=False, distance=None,
                   pixel_size=[1, 1], wavelength=1, weights=[100, 0, 0]):
    '''
    Simulate multiprobe tensor from the weighted directional derivatives of an initial probe.
    '''

    probe = simulate_probe(
        m, r_ratio, r_inner, propagate, distance, pixel_size, wavelength
    )
    grad_x, grad_y = t.gradient(probe)

    # Rescale the gradients to match the initial probe's modulus
    probe_area = probe.shape[0] * probe.shape[1]
    probe_scale = t.abs(probe).sum() / probe_area
    x_scale = t.abs(grad_x).sum() / probe_area
    y_scale = t.abs(grad_y).sum() / probe_area

    grad_x = grad_x * probe_scale / x_scale
    grad_y = grad_y * probe_scale / y_scale

    return t.stack([weights[0]*probe, weights[1]*grad_x, weights[2]*grad_y])


def simulate_multiprobe_defocus(m, r_ratio, r_inner=0, propagate=False, propa_distances=None,
                   pixel_size=[1, 1], wavelength=1):
    '''
    Simulate multiprobe tensor from the different focal distances of the same probe.
    '''
    
    multiprobes = [
        simulate_probe(m, r_ratio, r_inner, propagate, d, pixel_size, wavelength)
        for d in propa_distances
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
    '''
    Calculate the intensity of a wave.
    '''
    
    return t.abs(t.fft.fftshift(t.fft.fft2(t.fft.ifftshift(wave))))**2


def get_diffractions_direct(probe, obj, translations):
    '''
    Calculate diffraction pattern's intensity using ptycho_2D_round.
    '''

    if probe.dim() == 3: # multi-probe intensity
        multiprobe_interactions = t.stack([
            interactions.ptycho_2D_round(p, obj, translations) for p in probe
        ])
        diff_patterns = get_intensity(
            t.sum(multiprobe_interactions, 0)
        )
        
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
    
    
def AD_model_LBFGS(diffractions, sim_probe, obj_guess, translations, epochs,
             lr=1, patience=5, tolerance=1e-5, loss_f='PoissonNLL', show=False):
    '''
    Reconstruct the object using autodiff maximum likelihood.
    loss_f: {'PoissonNLL', 'MSE'}
    '''

    if loss_f == 'PoissonNLL': loss_fct = nn.PoissonNLLLoss(log_input=False)
    elif loss_f == 'MSE': loss_fct = nn.MSELoss()
        
    obj_guess.requires_grad = True
    optimizer = LBFGS([obj_guess], lr=lr)
    
    loss_list = []
    best_loss = float('inf')
    patience_counter = 0
    
    def closure_LBFGS():
        
        optimizer.zero_grad()
        
        predicted_patterns = get_diffractions_direct(
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


# def AD_model(diffractions, sim_probe, obj_guess, translations, epochs,
#              lr, batch_size, min_lr=1e-9, loss_f='PoissonNLL', show=False):
#     '''
#     Reconstruct the object using autodiff maximum likelihood.
#     loss_f: {'PoissonNLL', 'MSE'}
#     '''
    
#     if loss_f == 'PoissonNLL': loss_fct = nn.PoissonNLLLoss(log_input=False)
#     elif loss_f == 'MSE': loss_fct = nn.MSELoss()
        
#     obj_guess.requires_grad = True
#     optimizer = Adam([obj_guess], lr=lr)
#     scheduler = lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode='min', factor=0.1, patience=8, threshold=1e-4, eps=1e-10
#     )
    
#     loss_list = []
#     for e in range(epochs):

#         idx = t.randperm(diffractions.size(0))
        
#         l_batch = []
#         for b in range(translations.shape[0] // batch_size):
            
#             idx_batch = idx[b*batch_size:(b+1)*batch_size]
            
#             predicted_patterns = get_diffractions_direct(
#                 probe=sim_probe,
#                 obj=periodic_padding(obj_guess),
#                 translations=translations[idx_batch],
#             )

#             optimizer.zero_grad()
            
#             l = loss_fct(predicted_patterns, diffractions[idx_batch])
#             l_batch.append(l.item())
            
#             l.backward()
#             optimizer.step()
        
#         loss = sum(l_batch)
#         loss_list.append(loss)
#         scheduler.step(loss)
        
#         current_lr = scheduler.get_last_lr()[0]
        
#         if current_lr < lr:
#             batch_size = translations.shape[0]
            
#         if current_lr < min_lr:
#             print("Stopping early due to low learning rate")
#             break

#         if e%10 == 0 and show == True:
#             plt.imshow(t.abs(obj_guess).detach().to('cpu'))
#             plt.show(block=False)
#             plt.pause(0.001)

#         print(f'Epoch {e}: loss = {loss} | lr = {current_lr}')

#     return obj_guess.detach().clone().to('cpu'), loss_list

