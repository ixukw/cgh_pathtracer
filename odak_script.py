import odak
import torch


wavelength = 532e-9
pixel_pitch = 8e-6 
distance = 5e-3
propagation_type = 'Angular Spectrum'
k = odak.learn.wave.wavenumber(wavelength)


amplitude = torch.zeros(500, 500)
amplitude[200:300, 200:300 ] = 1.
phase = torch.randn_like(amplitude) * 2 * odak.pi
hologram = odak.learn.wave.generate_complex_field(amplitude, phase)


image_plane = odak.learn.wave.propagate_beam(
                                             hologram,
                                             k,
                                             distance,
                                             pixel_pitch,
                                             wavelength,
                                             propagation_type,
                                             zero_padding = [True, False, True]
                                            )
image_intensity = odak.learn.wave.calculate_amplitude(image_plane) ** 2 
odak.learn.tools.save_image(
                            'image_intensity.png', 
                            image_intensity, 
                            cmin = 0., 
                            cmax = 1.
                           )