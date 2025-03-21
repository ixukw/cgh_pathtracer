#include "render.h"
#include "intersection.h"
#include "material.h"
#include "parallel.h"
#include "path_tracing.h"
#include "vol_path_tracing.h"
#include "pcg.h"
#include "progress_reporter.h"
#include "scene.h"

#include </usr/local/include/fftw3.h>
#include <complex>

/// Render auxiliary buffers e.g., depth.
Image3 aux_render(const Scene &scene) {
    int w = scene.camera.width, h = scene.camera.height;
    Image3 img(w, h);

    constexpr int tile_size = 16;
    int num_tiles_x = (w + tile_size - 1) / tile_size;
    int num_tiles_y = (h + tile_size - 1) / tile_size;

    parallel_for([&](const Vector2i &tile) {
        int x0 = tile[0] * tile_size;
        int x1 = min(x0 + tile_size, w);
        int y0 = tile[1] * tile_size;
        int y1 = min(y0 + tile_size, h);
        for (int y = y0; y < y1; y++) {
            for (int x = x0; x < x1; x++) {
                Ray ray = sample_primary(scene.camera, Vector2((x + Real(0.5)) / w, (y + Real(0.5)) / h));
                RayDifferential ray_diff = init_ray_differential(w, h);
                if (std::optional<PathVertex> vertex = intersect(scene, ray, ray_diff)) {
                    Real dist = distance(vertex->position, ray.org);
                    Vector3 color{0, 0, 0};
                    if (scene.options.integrator == Integrator::Depth) {
                        color = Vector3{dist, dist, dist};
                    } else if (scene.options.integrator == Integrator::ShadingNormal) {
                        // color = (vertex->shading_frame.n + Vector3{1, 1, 1}) / Real(2);
                        color = vertex->shading_frame.n;
                    } else if (scene.options.integrator == Integrator::MeanCurvature) {
                        Real kappa = vertex->mean_curvature;
                        color = Vector3{kappa, kappa, kappa};
                    } else if (scene.options.integrator == Integrator::RayDifferential) {
                        color = Vector3{ray_diff.radius, ray_diff.spread, Real(0)};
                    } else if (scene.options.integrator == Integrator::MipmapLevel) {
                        const Material &mat = scene.materials[vertex->material_id];
                        const TextureSpectrum &texture = get_texture(mat);
                        auto *t = std::get_if<ImageTexture<Spectrum>>(&texture);
                        if (t != nullptr) {
                            const Mipmap3 &mipmap = get_img3(scene.texture_pool, t->texture_id);
                            Vector2 uv{modulo(vertex->uv[0] * t->uscale, Real(1)),
                                       modulo(vertex->uv[1] * t->vscale, Real(1))};
                            // ray_diff.radius stores approximatedly dpdx,
                            // but we want dudx -- we get it through
                            // dpdx / dpdu
                            Real footprint = vertex->uv_screen_size;
                            Real scaled_footprint = max(get_width(mipmap), get_height(mipmap)) *
                                                    max(t->uscale, t->vscale) * footprint;
                            Real level = log2(max(scaled_footprint, Real(1e-8f)));
                            color = Vector3{level, level, level};
                        }
                    }
                    img(x, y) = color;
                } else {
                    img(x, y) = Vector3{0, 0, 0};
                }
            }
        }
    }, Vector2i(num_tiles_x, num_tiles_y));

    return img;
}

Image3 path_render(const Scene &scene) {
    int w = scene.camera.width, h = scene.camera.height;
    Image3 img(w, h);

    constexpr int tile_size = 16;
    int num_tiles_x = (w + tile_size - 1) / tile_size;
    int num_tiles_y = (h + tile_size - 1) / tile_size;

    ProgressReporter reporter(num_tiles_x * num_tiles_y);
    parallel_for([&](const Vector2i &tile) {
        // Use a different rng stream for each thread.
        pcg32_state rng = init_pcg32(tile[1] * num_tiles_x + tile[0]);
        int x0 = tile[0] * tile_size;
        int x1 = min(x0 + tile_size, w);
        int y0 = tile[1] * tile_size;
        int y1 = min(y0 + tile_size, h);
        for (int y = y0; y < y1; y++) {
            for (int x = x0; x < x1; x++) {
                Spectrum radiance = make_zero_spectrum();
                int spp = scene.options.samples_per_pixel;
                for (int s = 0; s < spp; s++) {
                    radiance += path_tracing(scene, x, y, rng);
                }
                img(x, y) = radiance / Real(spp);
            }
        }
        reporter.update(1);
    }, Vector2i(num_tiles_x, num_tiles_y));
    reporter.done();
    return img;
}

Image3 compute_hologram(fftw_complex* E, Real w, Real h, Real lambda, Real z, Real p) {

    // Step 1: Apply the quadratic phase factor in real space
    for (int u = 0; u < w; u++) {
        for (int v = 0; v < h; v++) {
            int idx = v*w+u;
            //int idx = u*w+v;
            double up = (u - w / 2) * (lambda * z / (w * p));
            double vp = (v - h / 2) * (lambda * z / (h * p));
            double phase_factor = (M_PI / (lambda * z)) * (up * up + vp * vp);
            double real_part = cos(phase_factor);
            double imag_part = sin(phase_factor);

            // Multiply E by phase factor
            double E_real = E[idx][0];
            double E_imag = E[idx][1];
            //std::cout << "E" << E_real << ", " << E_imag << std::endl;
            E[idx][0] = E_real * real_part - E_imag * imag_part;
            E[idx][1] = E_real * imag_part + E_imag * real_part;
        }
    }

    // Step 2: Perform the 2D FFT to compute H(xi, eta)
    fftw_complex* H = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * w * h);
    fftw_plan plan = fftw_plan_dft_2d(w, h, E, H, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    // Step 3: Apply the phase factor and normalization to H
    for (int xi = 0; xi < w; xi++) {
        for (int eta = 0; eta < h; eta++) {
            int idx = eta * w + xi;
            //int idx = xi * w + eta;
            double xi_val = (xi - w / 2);
            double eta_val = (eta - h / 2);
            double phase_factor = (M_PI / (lambda * z)) * (xi_val * xi_val + eta_val * eta_val);
            double real_part = cos(phase_factor);
            double imag_part = sin(phase_factor);
            double H_real = H[idx][0];
            double H_imag = H[idx][1];

            // Apply phase factor and normalization
            if (lambda * z == 0) {
                std::cout << "PROBLEM" << std::endl;

            }
            if (isnan(phase_factor)) {
                std::cout << "PHASE FACTOR NAN" << std::endl;
            }
            H[idx][0] = (H_real * real_part - H_imag * imag_part) / (lambda * z);
            H[idx][1] = (H_real * imag_part + H_imag * real_part) / (lambda * z);
            //std::cout << H[idx][0] << H[idx][1] << std::endl;
        }
    }

    // Step 4: Compute the phase of H and store it in the phase array
    Image3 phase(w,h);
    for (int xi = 0; xi < w; xi++) {
        for (int eta = 0; eta < h; eta++) {
            //int idx = xi * w + eta;
            int idx = eta * w + xi;
            double real_part = H[idx][0];
            double imag_part = H[idx][1];

            //std::cout << real_part << ", " << imag_part << std::endl;
            
            Real val = (atan2(imag_part, real_part) + M_PI) / (2 * M_PI); // normalize to 0 to 1
            phase(xi,eta) = make_const_spectrum(val);
            //std::cout << phase(xi,eta) << std::endl;
        }
    }

    // Clean up
    fftw_free(H);
    return phase;
}

Image3 vol_path_render(const Scene &scene) {
    int w = scene.camera.width, h = scene.camera.height;
    Image3 img(w, h);

    fftw_complex* E = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * w * h);
    // Real wavelength = 0.632e-6; // wavelength in meters
    // Real z = 0.1;  // Propagation distance in meters
    // Real p = 1e-5;  // Pixel pitch in meters
    Real wavelength = 532e-9;
    Real z = 20e-1;
    Real p = 8e-6;

    //double* phase = new double[w * h];

    constexpr int tile_size = 16; // tiling does not seem to affect anything
    int num_tiles_x = (w + tile_size - 1) / tile_size;
    int num_tiles_y = (h + tile_size - 1) / tile_size;

    auto f = vol_path_tracing_3; // force the last one or 3 because i didnt modify 1,2,4,5
    // if (scene.options.vol_path_version == 1) {
    //     f = vol_path_tracing_1;
    // } else if (scene.options.vol_path_version == 2) {
    //     f = vol_path_tracing_2;
    // } else if (scene.options.vol_path_version == 3) {
    //     //f = vol_path_tracing_3;
    // } else if (scene.options.vol_path_version == 4) {
    //     f = vol_path_tracing_4;
    // } else if (scene.options.vol_path_version == 5) {
    //     f = vol_path_tracing_5;
    // } else if (scene.options.vol_path_version == 6) {
    //     f = vol_path_tracing;
    // }

    ProgressReporter reporter(num_tiles_x * num_tiles_y);
    parallel_for([&](const Vector2i &tile) {
        // Use a different rng stream for each thread.
        pcg32_state rng = init_pcg32(tile[1] * num_tiles_x + tile[0]);
        int x0 = tile[0] * tile_size;
        int x1 = min(x0 + tile_size, w);
        int y0 = tile[1] * tile_size;
        int y1 = min(y0 + tile_size, h);
        for (int y = y0; y < y1; y++) {
            for (int x = x0; x < x1; x++) { // for every pixel in image plane 
                Spectrum radiance = make_zero_spectrum();
                Real r_n = 0;
                int spp = scene.options.samples_per_pixel;
                for (int s = 0; s < spp; s++) { // sampling pixel density
                    PathVertex vertex;
                    Spectrum L = f(scene, x, y, rng, &vertex);
                    if (isfinite(L)) {
                        // Hacky: exclude NaNs in the rendering.
                        radiance += L;
                        
                        r_n += sqrt(pow(vertex.position.x-x,2) + pow(vertex.position.y-y,2)+vertex.position.z*vertex.position.z);
                        
                        Real val = 2 * M_PI / wavelength * r_n;
                        if (r_n == 0) {
                            r_n = 1e-6;
                        }
                        Real val2 = radiance[0] / r_n; // just use R channel for now
                        Real complex_field_real = val2 * cos(val);
                        Real complex_field_imag = val2 * sin(val);
                        E[y*w+x][0] += complex_field_real;
                        E[y*w+x][1] += complex_field_imag;
                    }
                }
            }
        }

         reporter.update(1);
    }, Vector2i(num_tiles_x, num_tiles_y));
    reporter.done();
    //return img;
    return compute_hologram(E, w, h, wavelength, z, p);
}


Image3 render(const Scene &scene) {
    if (scene.options.integrator == Integrator::Depth ||
            scene.options.integrator == Integrator::ShadingNormal ||
            scene.options.integrator == Integrator::MeanCurvature ||
            scene.options.integrator == Integrator::RayDifferential ||
            scene.options.integrator == Integrator::MipmapLevel) {
        return aux_render(scene);
    } else if (scene.options.integrator == Integrator::Path) {
        return path_render(scene);
    } else if (scene.options.integrator == Integrator::VolPath) {
        return vol_path_render(scene);
    } else {
        assert(false);
        return Image3();
    }
}
