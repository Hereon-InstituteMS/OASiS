# Probe of every catalog template

_Generated 2026-06-12 12:47:48 on this machine._

**Summary** — 85 templates probed: 85 generate-OK, 64 validate-OK, 63 run-completed (21 not probed at this stage), 37 produced a VTU (22 not probed at this stage).


## dealii — 39 templates (39 gen, 39 val, 38 run, 20 vtu)

| key | physics | variant | gen | val | run | vtu | fields | time | error |
|-----|---------|---------|-----|-----|-----|-----|--------|------|-------|
| `advection_dg_2d` | advection_dg | 2d | ✓ | ✓ | completed | ✗ |  | 2.1s |  |
| `cg_dg_coupled_2d` | cg_dg_coupled | 2d | ✓ | ✓ | completed | ✗ |  | 2.2s |  |
| `compressible_euler_2d` | compressible_euler | 2d | ✓ | ✓ | completed | ✗ |  | 2.1s |  |
| `contact_2d` | contact | 2d | ✓ | ✓ | completed | ✗ |  | 2.2s |  |
| `convection_diffusion_2d` | convection_diffusion | 2d | ✓ | ✓ | completed | ✓ | solution | 7.9s |  |
| `dg_advection_reaction_2d` | dg_advection_reaction | 2d | ✓ | ✓ | completed | ✗ |  | 2.2s |  |
| `dg_transport_2d` | dg_transport | 2d | ✓ | ✓ | completed | ✓ | solution | 8.2s |  |
| `eigenvalue_2d` | eigenvalue | 2d | ✓ | ✓ | completed | ✓ | eigenmode_0 | 6.8s |  |
| `error_estimation_2d` | error_estimation | 2d | ✓ | ✓ | completed | ✗ |  | 2.2s |  |
| `heat_2d_steady` | heat | 2d_steady | ✓ | ✓ | completed | ✓ | temperature | 6.7s |  |
| `heat_2d_transient` | heat | 2d_transient | ✓ | ✓ | completed | ✓ | temperature | 6.6s |  |
| `heat_rectangle` | heat | rectangle | ✓ | ✓ | completed | ✓ | temperature | 6.9s |  |
| `helmholtz_2d` | helmholtz | 2d | ✓ | ✓ | completed | ✓ | u_real, u_imag | 8.4s |  |
| `hp_adaptive_2d` | hp_adaptive | 2d | ✓ | ✓ | completed | ✓ | solution, fe_degree | 7.7s |  |
| `hyperelasticity_3d` | hyperelasticity | 3d | ✓ | ✓ | completed | ✓ | displacement | 65.7s |  |
| `linear_elasticity_2d` | linear_elasticity | 2d | ✓ | ✓ | completed | ✓ | ux, uy | 6.9s |  |
| `linear_elasticity_thick_beam` | linear_elasticity | thick_beam | ✓ | ✓ | completed | ✓ | ux, uy | 7.4s |  |
| `matrix_free_2d` | matrix_free | 2d | ✓ | ✓ | completed | ✗ |  | 2.4s |  |
| `mixed_laplacian_2d` | mixed_laplacian | 2d | ✓ | ✓ | completed | ✗ |  | 2.3s |  |
| `multigrid_2d` | multigrid | 2d | ✓ | ✓ | completed | ✗ |  | 2.4s |  |
| `multiphysics_dealii_2d` | multiphysics_dealii | 2d | ✓ | ✓ | completed | ✗ |  | 2.5s |  |
| `navier_stokes_2d` | navier_stokes | 2d | ✓ | ✓ | completed | ✗ |  | 4.1s |  |
| `nonlinear_2d_minimal_surface` | nonlinear | 2d_minimal_surface | ✓ | ✓ | completed | ✓ | solution | 7.4s |  |
| `nonlinear_elasticity_3d` | nonlinear_elasticity | 3d | ✓ | ✓ | completed | ✓ | displacement | 73.0s |  |
| `obstacle_problem_2d` | obstacle_problem | 2d | ✓ | ✓ | completed | ✗ |  | 2.7s |  |
| `optimal_control_2d` | optimal_control | 2d | ✓ | ✓ | completed | ✗ |  | 2.6s |  |
| `parallel_poisson_2d` | parallel_poisson | 2d | ✓ | ✓ | failed | — |  | 4.7s | Compilation failed: arks/probe_results/work/dealii/parallel_poisson_2d/main.cpp: |
| `phase_field_2d` | phase_field | 2d | ✓ | ✓ | completed | ✗ |  | 2.8s |  |
| `poisson_2d` | poisson | 2d | ✓ | ✓ | completed | ✓ | solution | 7.8s |  |
| `poisson_2d_adaptive` | poisson | 2d_adaptive | ✓ | ✓ | completed | ✓ |  | 7.8s |  |
| `poisson_3d` | poisson | 3d | ✓ | ✓ | completed | ✓ | solution | 8.3s |  |
| `poisson_l_domain` | poisson | l_domain | ✓ | ✓ | completed | ✓ | solution | 7.8s |  |
| `poisson_rectangle` | poisson | rectangle | ✓ | ✓ | completed | ✓ | solution | 8.7s |  |
| `stokes_2d` | stokes | 2d | ✓ | ✓ | completed | ✓ | velocity, pressure | 7.6s |  |
| `time_dependent_heat_2d` | time_dependent_heat | 2d | ✓ | ✓ | completed | ✗ |  | 2.6s |  |
| `time_dependent_ns_2d` | time_dependent_ns | 2d | ✓ | ✓ | completed | ✗ |  | 2.6s |  |
| `time_dependent_wave_2d` | time_dependent_wave | 2d | ✓ | ✓ | completed | ✗ |  | 2.7s |  |
| `topology_opt_dealii_2d` | topology_opt_dealii | 2d | ✓ | ✓ | completed | ✗ |  | 2.7s |  |
| `wave_2d` | wave | 2d | ✓ | ✓ | completed | ✓ | displacement | 8.1s |  |

## fourc — 46 templates (46 gen, 25 val, 25 run, 17 vtu)

| key | physics | variant | gen | val | run | vtu | fields | time | error |
|-----|---------|---------|-----|-----|-----|-----|--------|------|-------|
| `ale/ale_2d` | ale | ale_2d | ✓ | ✓ | completed | ✗ |  | 0.8s |  |
| `arterial_network/single_artery_1d` | arterial_network | single_artery_1d | ✓ | ✗ | — | — |  |  | Missing MATERIALS section |
| `beam_interaction/beam_contact_3d` | beam_interaction | beam_contact_3d | ✓ | ✗ | — | — |  |  | Missing MATERIALS section |
| `beam_interaction/beam_solid_meshtying_3d` | beam_interaction | beam_solid_meshtying_3d | ✓ | ✗ | — | — |  |  | Missing MATERIALS section |
| `beams/cantilever_dynamic` | beams | cantilever_dynamic | ✓ | ✓ | completed | ✓ |  | 0.8s | vtu-read ModuleNotFoundError: No module named 'pyvista' |
| `beams/cantilever_static` | beams | cantilever_static | ✓ | ✓ | completed | ✓ |  | 0.9s | vtu-read ModuleNotFoundError: No module named 'pyvista' |
| `brownian_dynamics/brownian_3d` | brownian_dynamics | brownian_3d | ✓ | ✗ | — | — |  |  | Missing MATERIALS section |
| `cardiac_monodomain/monodomain_3d` | cardiac_monodomain | monodomain_3d | ✓ | ✗ | — | — |  |  | Missing MATERIALS section |
| `cardiovascular0d/windkessel_3d` | cardiovascular0d | windkessel_3d | ✓ | ✓ | completed | ✗ |  | 1.2s |  |
| `constraint/constraint_3d` | constraint | constraint_3d | ✓ | ✓ | completed | ✓ |  | 0.8s | vtu-read ModuleNotFoundError: No module named 'pyvista' |
| `contact/penalty_3d` | contact | penalty_3d | ✓ | ✓ | completed | ✓ |  | 5.0s | vtu-read ModuleNotFoundError: No module named 'pyvista' |
| `ehl/ehl_3d` | ehl | ehl_3d | ✓ | ✗ | — | — |  |  | Missing MATERIALS section |
| `electrochemistry/nernst_planck_3d` | electrochemistry | nernst_planck_3d | ✓ | ✓ | completed | ✓ |  | 1.2s | vtu-read ModuleNotFoundError: No module named 'pyvista' |
| `fbi/penalty_3d` | fbi | penalty_3d | ✓ | ✗ | — | — |  |  | Missing MATERIALS section |
| `fluid/cavity_2d` | fluid | cavity_2d | ✓ | ✓ | completed | ✗ |  | 12.6s |  |
| `fluid/channel_2d` | fluid | channel_2d | ✓ | ✓ | completed | ✗ |  | 13.2s |  |
| `fluid_turbulence/les_channel_3d` | fluid_turbulence | les_channel_3d | ✓ | ✗ | — | — |  |  | Missing MATERIALS section |
| `fpsi/monolithic_3d` | fpsi | monolithic_3d | ✓ | ✗ | — | — |  |  | Missing MATERIALS section |
| `fs3i/fs3i_3d` | fs3i | fs3i_3d | ✓ | ✗ | — | — |  |  | Missing MATERIALS section |
| `fsi/fsi_2d` | fsi | fsi_2d | ✓ | ✓ | completed | ✗ |  | 3.2s |  |
| `fsi_xfem/xfem_fsi_3d` | fsi_xfem | xfem_fsi_3d | ✓ | ✗ | — | — |  |  | Missing MATERIALS section |
| `level_set/advection_2d` | level_set | advection_2d | ✓ | ✓ | completed | ✓ |  | 1.5s | vtu-read ModuleNotFoundError: No module named 'pyvista' |
| `low_mach/heated_channel_2d` | low_mach | heated_channel_2d | ✓ | ✓ | completed | ✓ |  | 1.0s | vtu-read ModuleNotFoundError: No module named 'pyvista' |
| `lubrication/slider_bearing_2d` | lubrication | slider_bearing_2d | ✓ | ✓ | completed | ✗ |  | 0.8s |  |
| `membrane/membrane_2d` | membrane | membrane_2d | ✓ | ✓ | completed | ✓ |  | 1.0s | vtu-read ModuleNotFoundError: No module named 'pyvista' |
| `mixture/mixture_3d` | mixture | mixture_3d | ✓ | ✓ | completed | ✓ |  | 0.9s | vtu-read ModuleNotFoundError: No module named 'pyvista' |
| `multiscale/fe2_3d` | multiscale | fe2_3d | ✓ | ✗ | — | — |  |  | Missing MATERIALS section |
| `particle_pd/plate_2d` | particle_pd | plate_2d | ✓ | ✗ | — | — |  |  | Missing MATERIALS section |
| `particle_sph/poiseuille_2d` | particle_sph | poiseuille_2d | ✓ | ✗ | — | — |  |  | Missing MATERIALS section |
| `pasi/dem_impact_3d` | pasi | dem_impact_3d | ✓ | ✗ | — | — |  |  | Missing MATERIALS section |
| `porous_media/single_phase_3d` | porous_media | single_phase_3d | ✓ | ✓ | completed | ✓ |  | 0.9s | vtu-read ModuleNotFoundError: No module named 'pyvista' |
| `reduced_airways/airways_1d` | reduced_airways | airways_1d | ✓ | ✗ | — | — |  |  | Missing MATERIALS section |
| `reduced_lung/lung_1d` | reduced_lung | lung_1d | ✓ | ✗ | — | — |  |  | Missing MATERIALS section |
| `scalar_transport/heat_transient_2d` | scalar_transport | heat_transient_2d | ✓ | ✓ | completed | ✓ |  | 0.9s | vtu-read ModuleNotFoundError: No module named 'pyvista' |
| `scalar_transport/poisson_2d` | scalar_transport | poisson_2d | ✓ | ✓ | completed | ✓ |  | 0.7s | vtu-read ModuleNotFoundError: No module named 'pyvista' |
| `shell/shell_3d` | shell | shell_3d | ✓ | ✓ | completed | ✓ |  | 5.1s | vtu-read ModuleNotFoundError: No module named 'pyvista' |
| `solid_mechanics/linear_2d` | solid_mechanics | linear_2d | ✓ | ✓ | completed | ✓ |  | 0.8s | vtu-read ModuleNotFoundError: No module named 'pyvista' |
| `solid_mechanics/nonlinear_3d` | solid_mechanics | nonlinear_3d | ✓ | ✓ | completed | ✓ |  | 0.7s | vtu-read ModuleNotFoundError: No module named 'pyvista' |
| `ssi/monolithic_elch_3d` | ssi | monolithic_elch_3d | ✓ | ✗ | — | — |  |  | Missing MATERIALS section |
| `ssti/monolithic_3d` | ssti | monolithic_3d | ✓ | ✗ | — | — |  |  | Missing MATERIALS section |
| `sti/monolithic_3d` | sti | monolithic_3d | ✓ | ✗ | — | — |  |  | Missing MATERIALS section |
| `structural_dynamics/genalpha_2d` | structural_dynamics | genalpha_2d | ✓ | ✓ | completed | ✓ |  | 1.1s | vtu-read ModuleNotFoundError: No module named 'pyvista' |
| `thermo/thermo_2d` | thermo | thermo_2d | ✓ | ✓ | completed | ✗ |  | 0.7s |  |
| `thermo/thermo_3d` | thermo | thermo_3d | ✓ | ✓ | completed | ✗ |  | 0.8s |  |
| `tsi/monolithic_3d` | tsi | monolithic_3d | ✓ | ✓ | completed | ✓ |  | 5.8s | vtu-read ModuleNotFoundError: No module named 'pyvista' |
| `xfem_fluid/xfem_3d` | xfem_fluid | xfem_3d | ✓ | ✗ | — | — |  |  | Missing MATERIALS section |