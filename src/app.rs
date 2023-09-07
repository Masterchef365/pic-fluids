/// We derive Deserialize/Serialize so we can persist app state on shutdown.
pub struct TemplateApp {
    // Sim state
    sim: Sim,

    // Settings
    dt: f32,
    solver_iters: usize,
    stiffness: f32,
    gravity: f32,
    solver: IncompressibilitySolver,

    width: usize,
    height: usize,
    n_particles: usize,
    calc_rest_density_from_radius: bool,
    show_arrows: bool,
    show_grid: bool,
    grid_vel_scale: f32,
    pause: bool,
    single_step: bool,
    pic_apic_ratio: f32,
    n_colors: usize,
    enable_incompress: bool,
    enable_particle_collisions: bool,
    enable_grid_transfer: bool,
    saturation: f32,
    rand_std_dev: f32,

    well: bool,
    source_color_idx: ParticleType,
    source_rate: usize,

    show_settings_only: bool,

    advanced: bool,
}

impl TemplateApp {
    /// Called once before the first frame.
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let (width, height) = if is_mobile(&cc.egui_ctx) {
            (70, 150)
        } else {
            (120, 80)
        };
        let n_particles = 4_000;
        let particle_radius = 0.16;

        let n_colors = 3;
        let rand_std_dev = 10.;
        let life = LifeConfig::random(n_colors, rand_std_dev);
        let sim = Sim::new(width, height, n_particles, particle_radius, life);

        Self {
            rand_std_dev,
            saturation: 1./100.,
            enable_grid_transfer: true,
            enable_particle_collisions: false,
            enable_incompress: true,
            advanced: false,
            n_colors,
            source_rate: 0,
            pic_apic_ratio: 1.,
            calc_rest_density_from_radius: true,
            single_step: false,
            dt: 0.02,
            solver_iters: 15,
            stiffness: 0.,
            gravity: 9.8,
            sim,
            width,
            height,
            n_particles,
            solver: IncompressibilitySolver::GaussSeidel,
            well: false,
            source_color_idx: 0,
            show_arrows: false,
            pause: false,
            grid_vel_scale: 0.05,
            show_grid: false,
            show_settings_only: false,
        }
    }
}

fn is_mobile(ctx: &egui::Context) -> bool {
    matches!(ctx.os(), OperatingSystem::Android | OperatingSystem::IOS)
}

impl eframe::App for TemplateApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.set_visuals(egui::Visuals::dark());

        // Update continuously
        ctx.request_repaint();
        if is_mobile(ctx) {
            CentralPanel::default().show(ctx, |ui| {
                ui.checkbox(&mut self.show_settings_only, "Show settings");
                if self.show_settings_only {
                    ScrollArea::both().show(ui, |ui| self.settings_gui(ui));
                } else {
                    Frame::canvas(ui.style()).show(ui, |ui| self.sim_widget(ui));
                }
            });
        } else {
            SidePanel::left("Settings").show(ctx, |ui| {
                ScrollArea::both().show(ui, |ui| self.settings_gui(ui))
            });
            CentralPanel::default().show(ctx, |ui| {
                Frame::canvas(ui.style()).show(ui, |ui| self.sim_widget(ui))
            });
        }
    }
}

use crate::array2d::{Array2D, GridPos};
use crate::query_accel::QueryAccelerator;
use eframe::egui::{DragValue, Grid, Rgba, RichText, ScrollArea, Slider, Ui};
use egui::os::OperatingSystem;
use egui::SidePanel;
use egui::{CentralPanel, Frame, Rect, Sense};
use glam::Vec2;
use rand::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IncompressibilitySolver {
    Jacobi,
    GaussSeidel,
}

impl TemplateApp {
    fn update(&mut self) {
        // Update
        if !self.pause || self.single_step {
            for _ in 0..self.source_rate {
                let pos = Vec2::new(10., 90.);
                let vel = Vec2::ZERO;
                self.sim.particles.push(Particle {
                    pos,
                    vel,
                    deriv: [Vec2::ZERO; 2],
                    color: self.source_color_idx,
                });
            }

            if self.well {
                for part in &mut self.sim.particles {
                    if part.pos.x < 20. {
                        part.vel += self.dt * 9.;
                    }
                }
            }

            if self.calc_rest_density_from_radius {
                self.sim.rest_density = calc_rest_density(self.sim.particle_radius);
            }

            self.sim.step(
                self.dt,
                self.solver_iters,
                self.stiffness,
                self.gravity,
                self.pic_apic_ratio,
                self.solver,
                self.enable_incompress,
                self.enable_particle_collisions,
                self.enable_grid_transfer,
                self.saturation
            );

            self.single_step = false;
        }

        // Display
        /*
        io.send(&UploadMesh {
            mesh: particles_mesh(&self.sim.particles, &self.sim.life),
            id: POINTS_RDR,
        });

        let mut lines = Mesh::new();
        if self.show_arrows {
            draw_grid_arrows(&mut lines, &self.sim.grid, self.grid_vel_scale);
        }
        if self.show_grid {
            draw_grid(&mut lines, &self.sim.grid)
        }

        io.send(&UploadMesh {
            mesh: lines,
            id: LINES_RDR,
        });
        */
    }

    fn sim_widget(&mut self, ui: &mut Ui) {
        let (rect, response) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());

        let coords = CoordinateMapping::new(&self.sim.grid, rect);

        // Move particles
        move_particles_from_egui(&mut self.sim.particles, 4., &coords, self.dt, response);

        // Step particles
        if !self.pause || self.single_step {
            self.update();
            self.single_step = false;
        }

        // Draw particles
        let painter = ui.painter_at(rect);
        let radius = coords
            .sim_to_egui_vect(Vec2::splat(self.sim.particle_radius))
            .length();

        for part in &self.sim.particles {
            let color = self.sim.life.colors[part.color as usize];
            painter.circle_filled(
                coords.sim_to_egui(part.pos) + rect.left_top().to_vec2(),
                1.,
                //radius,
                color_to_egui(color),
            );
        }
    }

    fn settings_gui(&mut self, ui: &mut Ui) {
        let mut reset = false;
        ui.separator();
        ui.strong("Simulation state");

        if ui
            .add(
                DragValue::new(&mut self.n_particles)
                    .prefix("# of particles: ")
                    .clamp_range(1..=usize::MAX)
                    .speed(4),
            )
            .changed()
        {
            let mut rng = rand::thread_rng();
            self.sim.particles.resize_with(self.n_particles, || {
                random_particle(
                    &mut rng,
                    self.sim.grid.width(),
                    self.sim.grid.height(),
                    &self.sim.life,
                )
            });
        }
        if ui
            .add(
                DragValue::new(&mut self.n_colors)
                    .prefix("# of colors: ")
                    .clamp_range(1..=255),
            )
            .changed()
        {
            let old_size = self.sim.life.behaviours.width();
            let mut new_behav_array = Array2D::new(self.n_colors, self.n_colors);
            for i in 0..self.n_colors {
                for j in 0..self.n_colors {
                    if i < old_size && j < old_size {
                        new_behav_array[(i, j)] = self.sim.life.behaviours[(i, j)];
                    } else {
                        new_behav_array[(i, j)] = LifeConfig::random_behaviour(self.rand_std_dev);
                    }
                }
            }
            self.sim.life.behaviours = new_behav_array;

            self.sim
                .life
                .colors
                .resize_with(self.n_colors, || random_color(&mut rand::thread_rng()));
            reset = true;
        }
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.pause, "Pause");
            self.single_step |= ui.button("Step").clicked();
        });
        if ui.button("Reset").clicked() {
            reset = true;
        }
        if self.advanced {
            ui.horizontal(|ui| {
                ui.add(
                    DragValue::new(&mut self.width)
                        .prefix("Width: ")
                        .clamp_range(1..=usize::MAX),
                );
                ui.add(
                    DragValue::new(&mut self.height)
                        .prefix("Height: ")
                        .clamp_range(1..=usize::MAX),
                );
            });
        }

        ui.separator();
        ui.strong("Kinematics");
        ui.add(
            DragValue::new(&mut self.dt)
                .prefix("Δt (time step): ")
                .speed(1e-3),
        );
        ui.horizontal(|ui| {
            ui.add(
                DragValue::new(&mut self.gravity)
                    .prefix("Gravity: ")
                    .speed(1e-2),
            );
            if ui.button("Zero-G").clicked() {
                self.gravity = 0.;
            }
        });
        if self.advanced {
            ui.add(
                DragValue::new(&mut self.sim.damping)
                    .prefix("Damping: ")
                    .speed(1e-3),
            );
            ui.add(Slider::new(&mut self.pic_apic_ratio, 0.0..=1.0).text("PIC - APIC"));
        }

        ui.separator();
        ui.horizontal(|ui| {
            ui.strong("Particle collisions");
            ui.checkbox(&mut self.enable_particle_collisions, "");
        });
        ui.add(
            DragValue::new(&mut self.sim.particle_radius)
                .prefix("Particle radius: ")
                .speed(1e-2)
                .clamp_range(1e-2..=5.0),
        );
        if self.advanced {
            ui.horizontal(|ui| {
                ui.add(
                    DragValue::new(&mut self.sim.rest_density)
                        .prefix("Rest density: ")
                        .speed(1e-2),
                );
                ui.checkbox(&mut self.calc_rest_density_from_radius, "From radius");
            });
            ui.add(
                DragValue::new(&mut self.stiffness)
                    .prefix("Stiffness: ")
                    .speed(1e-2),
            );
        }

        if self.advanced {
            ui.separator();
            ui.horizontal(|ui| {
                ui.strong("Incompressibility Solver");
                ui.checkbox(&mut self.enable_incompress, "");
            });
            ui.checkbox(&mut self.enable_grid_transfer, "Grid transfer");
            ui.add(DragValue::new(&mut self.solver_iters).prefix("Solver iterations: "));
            ui.add(
                DragValue::new(&mut self.sim.over_relax)
                    .prefix("Over-relaxation: ")
                    .speed(1e-2)
                    .clamp_range(0.0..=1.95),
            );
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.solver, IncompressibilitySolver::Jacobi, "Jacobi");
                ui.selectable_value(
                    &mut self.solver,
                    IncompressibilitySolver::GaussSeidel,
                    "Gauss Seidel",
                );
            });
        }

        ui.separator();
        ui.strong("Particle source");
        ui.add(DragValue::new(&mut self.source_rate).prefix("Particle inflow rate: "));
        ui.horizontal(|ui| {
            ui.label("Particle inflow color: ");
            for (idx, &color) in self.sim.life.colors.iter().enumerate() {
                let color_marker = RichText::new("#####").color(color_to_egui(color));
                let button = ui.selectable_label(idx as u8 == self.source_color_idx, color_marker);
                if button.clicked() {
                    self.source_color_idx = idx as u8;
                }
            }
            self.source_color_idx = self
                .source_color_idx
                .min(self.sim.life.colors.len() as u8 - 1);
        });
        ui.checkbox(&mut self.well, "Particle well");

        ui.separator();
        ui.strong("Particle life");
        let mut behav_cfg = self.sim.life.behaviours[(0, 0)];

        ui.add(DragValue::new(&mut self.saturation).prefix("Saturation: ").speed(1e-4));
        if self.advanced {
            ui.add(
                DragValue::new(&mut behav_cfg.max_inter_dist)
                    .clamp_range(0.0..=14.0)
                    .speed(1e-2)
                    .prefix("Max interaction dist: "),
            );
            ui.add(
                DragValue::new(&mut behav_cfg.default_repulse)
                    .speed(1e-2)
                    .prefix("Default repulse: "),
            );
            ui.add(
                DragValue::new(&mut behav_cfg.inter_threshold)
                    .clamp_range(0.0..=14.0)
                    .speed(1e-2)
                    .prefix("Interaction threshold: "),
            );
        }
        for b in self.sim.life.behaviours.data_mut() {
            b.max_inter_dist = behav_cfg.max_inter_dist;
            b.inter_threshold = behav_cfg.inter_threshold;
            b.default_repulse = behav_cfg.default_repulse;
        }

        ui.label("Interactions:");
        Grid::new("Particle Life Grid").show(ui, |ui| {
            // Top row
            //ui.label("Life");
            ui.label("");
            for color in &mut self.sim.life.colors {
                ui.color_edit_button_rgb(color);
            }
            ui.end_row();

            // Grid
            let len = self.sim.life.colors.len();
            for (row_idx, color) in self.sim.life.colors.iter_mut().enumerate() {
                ui.color_edit_button_rgb(color);
                for column in 0..len {
                    let behav = &mut self.sim.life.behaviours[(column, row_idx)];
                    ui.add(DragValue::new(&mut behav.inter_strength).speed(1e-2));
                }
                ui.end_row();
            }
        });

        ui.horizontal(|ui| {
            if ui.button("Randomize behaviours").clicked() {
                self.sim.life = LifeConfig::random(self.n_colors, self.rand_std_dev);
                reset = true;
            }
            ui.add(DragValue::new(&mut self.rand_std_dev).prefix("σ: "))
        });
        if ui.button("Symmetric forces").clicked() {
            let n = self.sim.life.colors.len();
            for i in 0..n {
                for j in 0..i {
                    self.sim.life.behaviours[(i, j)] = self.sim.life.behaviours[(j, i)]
                }
            }
        }
        if ui.button("Lifeless").clicked() {
            self.sim
                .life
                .behaviours
                .data_mut()
                .iter_mut()
                .for_each(|b| b.inter_strength = 0.);
        }

        /*
        if self.advanced {
        ui.separator();
        ui.strong("Debug");
        ui.checkbox(&mut self.show_grid, "Show grid");
        ui.horizontal(|ui| {
        ui.checkbox(&mut self.show_arrows, "Show arrows");
        ui.add(
        DragValue::new(&mut self.grid_vel_scale)
        .prefix("Scale: ")
        .speed(1e-2)
        .clamp_range(0.0..=f32::INFINITY),
        )
        });
        }
        */

        if reset {
            let damp = self.sim.damping;
            self.sim = Sim::new(
                self.width,
                self.height,
                self.n_particles,
                self.sim.particle_radius,
                self.sim.life.clone(),
            );
            self.sim.damping = damp;
        }

        ui.checkbox(&mut self.advanced, "Advanced settings");
        ui.hyperlink_to(
            "GitHub repository",
            "https://github.com/Masterchef365/pic-fluids",
        );
    }
}

const SIM_TO_MODEL_DOWNSCALE: f32 = 100.;
fn simspace_to_modelspace(pos: Vec2) -> [f32; 3] {
    [
        (pos.x / SIM_TO_MODEL_DOWNSCALE) * 2. - 1.,
        0.,
        (pos.y / SIM_TO_MODEL_DOWNSCALE) * 2. - 1.,
    ]
}

/*
fn particles_mesh(particles: &[Particle], life: &LifeConfig) -> Shape {
Mesh {
vertices: particles
.iter()
.map(|p| Vertex::new(simspace_to_modelspace(p.pos), life.colors[p.color as usize]))
.collect(),
indices: (0..particles.len() as u32).collect(),
}
}
*/

#[derive(Clone)]
struct Sim {
    /// Particles
    particles: Vec<Particle>,
    /// Cell wall velocity, staggered grid
    grid: Array2D<GridCell>,
    /// Rest density, in particles/unit^2
    rest_density: f32,
    particle_radius: f32,
    over_relax: f32,
    life: LifeConfig,
    damping: f32,
}

#[derive(Copy, Clone, Debug, Default)]
struct GridCell {
    /// Flow rate through the top and left faces of this cell
    vel: Vec2,
    /// Pressure inside this cell
    pressure: f32,
}

#[derive(Copy, Clone, Debug, Default)]
struct Particle {
    /// Position
    pos: Vec2,
    /// Velocity
    vel: Vec2,
    /// Particle type
    color: ParticleType,
    /// Velocity derivatives
    deriv: [Vec2; 2],
}

fn calc_rest_density(particle_radius: f32) -> f32 {
    // Assume hexagonal packing
    let packing_density = std::f32::consts::PI / 2. / 3_f32.sqrt();
    let particle_area = std::f32::consts::PI * particle_radius.powi(2);

    // A guess for particle life
    packing_density / particle_area
}

fn random_particle(rng: &mut impl Rng, width: usize, height: usize, life: &LifeConfig) -> Particle {
    let pos = Vec2::new(
        rng.gen_range(1.0..=(width - 2) as f32),
        rng.gen_range(1.0..=(height - 2) as f32),
    );
    let color = rng.gen_range(0..life.colors.len() as u8);
    Particle {
        pos,
        vel: Vec2::ZERO,
        color,
        deriv: [Vec2::ZERO; 2],
    }
}

impl Sim {
    pub fn new(
        width: usize,
        height: usize,
        n_particles: usize,
        particle_radius: f32,
        life: LifeConfig,
    ) -> Self {
        // Uniformly placed, random particles
        let mut rng = rand::thread_rng();
        let particles = (0..n_particles)
            .map(|_| random_particle(&mut rng, width, height, &life))
            .collect();

        // Assume half-hexagonal packing density...
        //let rest_density = calc_rest_density(particle_radius);
        let rest_density = 2.8;

        Sim {
            damping: 0.,
            life,
            particles,
            grid: Array2D::new(width, height),
            rest_density,
            particle_radius,
            over_relax: 1.8,
        }
    }

    pub fn step(
        &mut self,
        dt: f32,
        solver_iters: usize,
        stiffness: f32,
        gravity: f32,
        pic_apic_ratio: f32,
        solver: IncompressibilitySolver,
        enable_incompress: bool,
        enable_particle_collisions: bool,
        enable_grid_transfer: bool,
        saturation: f32,
    ) {
        // Step particles
        apply_global_force(&mut self.particles, Vec2::new(0., -gravity), dt);
        particle_interactions(&mut self.particles, &mut self.life, dt, saturation);
        step_particles(&mut self.particles, dt, self.damping);
        if enable_particle_collisions {
            enforce_particle_radius(&mut self.particles, self.particle_radius);
        }
        enforce_particle_pos(&mut self.particles, &self.grid);

        // Step grid
        particles_to_grid(&self.particles, &mut self.grid, pic_apic_ratio);
        let solver_fn = match solver {
            IncompressibilitySolver::Jacobi => solve_incompressibility_jacobi,
            IncompressibilitySolver::GaussSeidel => solve_incompressibility_gauss_seidel,
        };

        if enable_incompress {
            solver_fn(
                &mut self.grid,
                solver_iters,
                self.rest_density,
                self.over_relax,
                stiffness,
            );
        }

        if enable_grid_transfer {
            grid_to_particles(&mut self.particles, &self.grid);
        }
    }
}

/// Move particles forwards in time by `dt`, assuming unit mass for all particles.
fn step_particles(particles: &mut [Particle], dt: f32, damping: f32) {
    for part in particles {
        part.vel *= 1. - damping;
        part.pos = part.pos + part.vel * dt;
    }
}

/// Apply a force to all particles, e.g. gravity
fn apply_global_force(particles: &mut [Particle], g: Vec2, dt: f32) {
    for part in particles {
        part.vel += g * dt;
    }
}

/// Offset from particles to U grid (the U grid is 0.5 units positive to where the particles sit)
const OFFSET_U: Vec2 = Vec2::new(0., 0.5);
/// Offset from particles to V grid
const OFFSET_V: Vec2 = Vec2::new(0.5, 0.);

/// Insert information such as velocity and pressure into the grid
fn particles_to_grid(particles: &[Particle], grid: &mut Array2D<GridCell>, pic_apic_ratio: f32) {
    // Clear the grid
    grid.data_mut()
        .iter_mut()
        .for_each(|c| *c = GridCell::default());

    // Accumulate velocity on grid
    // Here we abuse the pressure of each grid cell to by mass correctly
    for part in particles {
        let u_pos = part.pos - OFFSET_U;
        scatter(u_pos, grid, |c, n, w| {
            c.vel.x +=
                w * (part.vel.x + (index_to_pos(n) - u_pos).dot(part.deriv[0]) * pic_apic_ratio)
        });
        scatter(u_pos, grid, |c, _, w| c.pressure += w);
    }
    grid.data_mut().iter_mut().for_each(|c| {
        if c.pressure != 0.0 {
            c.vel.x /= c.pressure;
        }
    });
    grid.data_mut().iter_mut().for_each(|c| c.pressure = 0.);

    // And then we do again for u
    for part in particles {
        let v_pos = part.pos - OFFSET_V;
        scatter(v_pos, grid, |c, n, w| {
            c.vel.y +=
                w * (part.vel.y + (index_to_pos(n) - v_pos).dot(part.deriv[1]) * pic_apic_ratio)
        });
        scatter(v_pos, grid, |c, _, w| c.pressure += w);
    }
    grid.data_mut().iter_mut().for_each(|c| {
        if c.pressure != 0.0 {
            c.vel.y /= c.pressure;
        }
    });
    grid.data_mut().iter_mut().for_each(|c| c.pressure = 0.);

    // Now we actually set the pressure
    for part in particles {
        grid[grid_tl(part.pos)].pressure += 1.;
    }
}

/// Returns the weights for each grid
/// corner in the order [tl, tr, bl, br]
///
/// Panics if coordinates are inverted
fn weights(pos: Vec2) -> [f32; 4] {
    let right = pos.x.fract();
    let left = 1. - right;
    let bottom = pos.y.fract();
    let top = 1. - bottom;

    [top * left, top * right, bottom * left, bottom * right]
}

/// Returns the grid position corresponding to the top-left of this cell
fn grid_tl(pos: Vec2) -> GridPos {
    (pos.x.floor() as usize, pos.y.floor() as usize)
}

/// Returns the grid positions corresponding to [tl, tr, bl, br]
fn grid_neighborhood(_tl @ (i, j): GridPos) -> [GridPos; 4] {
    [(i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)]
}

/// Performs a weighted sum of the grid field chosen by f
fn gather<T>(pos: Vec2, grid: &Array2D<T>, f: fn(&T) -> f32) -> f32 {
    let weights = weights(pos);
    let neighbors = grid_neighborhood(grid_tl(pos));

    let mut total = 0.0;
    for (w, n) in weights.into_iter().zip(neighbors) {
        total += w * f(&grid[n]);
    }
    total
}

/// Performs a weighted sum of the grid field chosen by f
fn gather_vector(pos: Vec2, f: impl Fn(GridPos) -> Vec2) -> Vec2 {
    let neighbors = grid_neighborhood(grid_tl(pos));

    let mut total = Vec2::ZERO;
    for n in neighbors {
        total += f(n);
    }
    total
}

/// Performs a weighted accumulation on the grid field chosen by f
fn scatter<T>(pos: Vec2, grid: &mut Array2D<T>, mut f: impl FnMut(&mut T, GridPos, f32)) {
    let weights = weights(pos);
    let neighbors = grid_neighborhood(grid_tl(pos));

    for (w, n) in weights.into_iter().zip(neighbors) {
        f(&mut grid[n], n, w);
    }
}

/// Solve incompressibility on the grid cells, includinge contribution from presssure
fn solve_incompressibility_jacobi(
    grid: &mut Array2D<GridCell>,
    iterations: usize,
    rest_density: f32,
    overrelaxation: f32,
    stiffness: f32,
) {
    let mut tmp = grid.clone();

    for step in 0..iterations {
        for i in 0..grid.width() - 1 {
            for j in 0..grid.height() - 1 {
                let local_pressure = grid[(i, j)].pressure;
                let has_particles = local_pressure > 0.;

                let checkerboard = (i & 1) ^ (j & 1) ^ (step & 1);

                if checkerboard == 0 && has_particles {
                    let horiz_div = grid[(i + 1, j)].vel.x - grid[(i, j)].vel.x;
                    let vert_div = grid[(i, j + 1)].vel.y - grid[(i, j)].vel.y;
                    let total_div = horiz_div + vert_div;

                    let pressure_contrib = stiffness * (grid[(i, j)].pressure - rest_density);
                    let d = overrelaxation * total_div - pressure_contrib;
                    let d = d / 4.;

                    tmp[(i, j)].vel.x = grid[(i, j)].vel.x + d;
                    tmp[(i + 1, j)].vel.x = grid[(i + 1, j)].vel.x - d;

                    tmp[(i, j)].vel.y = grid[(i, j)].vel.y + d;
                    tmp[(i, j + 1)].vel.y = grid[(i, j + 1)].vel.y - d;
                }
            }
        }

        std::mem::swap(&mut tmp, grid);
        enforce_grid_boundary(grid);
    }
}

/// Solve incompressibility on the grid cells, includinge contribution from presssure
fn solve_incompressibility_gauss_seidel(
    grid: &mut Array2D<GridCell>,
    iterations: usize,
    rest_density: f32,
    overrelaxation: f32,
    stiffness: f32,
) {
    // TODO: Use Jacobi method instead!
    for _ in 0..iterations {
        for i in 0..grid.width() - 1 {
            for j in 0..grid.height() - 1 {
                if grid[(i, j)].pressure > 0. {
                    let horiz_div = grid[(i + 1, j)].vel.x - grid[(i, j)].vel.x;
                    let vert_div = grid[(i, j + 1)].vel.y - grid[(i, j)].vel.y;
                    let total_div = horiz_div + vert_div;

                    let pressure_contrib = stiffness * (grid[(i, j)].pressure - rest_density);
                    let d = overrelaxation * total_div - pressure_contrib;
                    let d = d / 4.;

                    grid[(i, j)].vel.x += d;
                    grid[(i + 1, j)].vel.x -= d;
                    grid[(i, j)].vel.y += d;
                    grid[(i, j + 1)].vel.y -= d;
                }
            }
        }

        enforce_grid_boundary(grid);
    }
}

fn grid_to_particles(particles: &mut [Particle], grid: &Array2D<GridCell>) {
    for part in particles {
        let u_pos = part.pos - OFFSET_U;
        let v_pos = part.pos - OFFSET_V;

        // Interpolate velocity onto particles
        part.vel = Vec2::new(
            gather(u_pos, grid, |c| c.vel.x),
            gather(v_pos, grid, |c| c.vel.y),
        );

        fn gradient(p: GridPos, v: Vec2) -> Vec2 {
            let p = index_to_pos(p);
            let f = v.fract();

            if v.x > p.x {
                if v.y > p.y {
                    Vec2::new(f.y - 1., f.x - 1.)
                } else {
                    Vec2::new(-f.y, 1. - f.x)
                }
            } else {
                if v.y > p.y {
                    Vec2::new(1. - f.y, -f.x)
                } else {
                    Vec2::new(f.y, f.x)
                }
            }
        }

        /*
        let v = Vec2::new(0.25, 0.15);
        dbg!(gradient((0, 0), v));
        dbg!(gradient((1, 0), v));
        dbg!(gradient((0, 1), v));
        dbg!(gradient((1, 1), v));
        todo!();
        */

        // Interpolate grid vectors
        part.deriv[0] = gather_vector(u_pos, |p| grid[p].vel.x * gradient(p, u_pos));
        part.deriv[1] = gather_vector(v_pos, |p| grid[p].vel.y * gradient(p, v_pos));
    }
}

fn enforce_particle_pos(particles: &mut [Particle], grid: &Array2D<GridCell>) {
    for part in particles {
        // Ensure particles are within the grid
        let min_x = 1.0;
        let max_x = (grid.width() - 2) as f32;
        let min_y = 1.0;
        let max_y = (grid.height() - 2) as f32;

        if part.pos.x < min_x {
            part.pos.x = min_x;
            part.vel.x *= -1.;
        }

        if part.pos.x > max_x {
            part.pos.x = max_x;
            part.vel.x *= -1.;
        }

        if part.pos.y < min_y {
            part.pos.y = min_y;
            part.vel.y *= -1.;
        }

        if part.pos.y > max_y {
            part.pos.y = max_y;
            part.vel.y *= -1.;
        }
    }
}

fn enforce_grid_boundary(grid: &mut Array2D<GridCell>) {
    let (w, h) = (grid.width(), grid.height());
    for y in 0..h {
        grid[(0, y)].vel.x = 0.0;
        grid[(w - 1, y)].vel.x = 0.0;
    }

    for x in 0..w {
        grid[(x, 0)].vel.y = 0.0;
        grid[(0, h - 1)].vel.y = 0.0;
    }
}

fn enforce_particle_radius(particles: &mut [Particle], radius: f32) {
    let mut points: Vec<Vec2> = particles.iter().map(|p| p.pos).collect();
    let mut accel = QueryAccelerator::new(&points, radius * 2.);
    //accel.stats("Collisions");

    //let mut rng = rand::thread_rng();

    let mut neigh = vec![];
    for i in 0..particles.len() {
        //neigh.clear();
        neigh.extend(accel.query_neighbors(&points, i, points[i]));

        for neighbor in neigh.drain(..) {
            let diff = points[neighbor] - points[i];
            let dist = diff.length();
            if dist > 0. {
                let norm = diff.normalize();
                let needed_dist = radius * 2. - dist;
                let prev_pos = points[i];
                let prev_neighbor = points[neighbor];
                points[i] -= norm * needed_dist / 2.;
                points[neighbor] += norm * needed_dist / 2.;
                accel.replace_point(i, prev_pos, points[i]);
                accel.replace_point(neighbor, prev_neighbor, points[neighbor]);
            }
        }
    }

    particles
        .iter_mut()
        .zip(&points)
        .for_each(|(part, point)| part.pos = *point);
}

fn particle_interactions(particles: &mut [Particle], cfg: &LifeConfig, dt: f32, saturation: f32) {
    let points: Vec<Vec2> = particles.iter().map(|p| p.pos).collect();
    let accel = QueryAccelerator::new(&points, cfg.max_interaction_radius());
    //accel.stats("Life");

    let mut neighbors = vec![];

    let mut accum = vec![0.; cfg.colors.len()];

    for i in 0..particles.len() {
        accum.fill(0.);

        neighbors.clear();
        neighbors.extend(accel.query_neighbors(&points, i, points[i]));

        let a = points[i];
        let i_color = particles[i].color;

        for &neighbor in &neighbors {
            let b = points[neighbor];
            let their_color = particles[neighbor].color;
            let behav = cfg.get_behaviour(i_color, their_color);
            let f = behav.force(a.distance(b));
            if f > 0. {
                accum[particles[neighbor].color as usize] += f;
            }
        }

        for &neighbor in &neighbors {
            let a = points[i];
            let b = points[neighbor];

            // The vector pointing from a to b
            let diff = b - a;

            // Distance is capped
            let dist = diff.length();
            if dist > 0. {
                // Accelerate towards b
                let normal = diff.normalize();
                let their_color = particles[neighbor].color;
                let behav = cfg.get_behaviour(i_color, their_color );
                let mut f = behav.force(dist);

                f -= accum[their_color as usize] * saturation;

                let accel = normal * f;

                particles[i].vel += accel * dt;
            }
        }
    }
}

/*
fn draw_arrow(mesh: &mut Mesh, pos: Vec2, dir: Vec2, color: [f32; 3], flanges: f32) {
let mut vertex = |pt: Vec2| mesh.push_vertex(Vertex::new(simspace_to_modelspace(pt), color));

let p1 = vertex(pos);

let end = pos + dir;
let p2 = vertex(end);

let angle = 0.3;
let f1 = vertex(end - flanges * dir.rotate(Vec2::from_angle(angle)));
let f2 = vertex(end - flanges * dir.rotate(Vec2::from_angle(-angle)));

mesh.push_indices(&[p1, p2, p2, f1, p2, f2]);
}

fn draw_grid_arrows(mesh: &mut Mesh, grid: &Array2D<GridCell>, vel_scale: f32) {
for i in 0..grid.width() {
for j in 0..grid.height() {
let c = grid[(i, j)];

if c.pressure == 0.0 {
continue;
}

let v = Vec2::new(i as f32, j as f32);

let flanges = 0.5;
draw_arrow(
mesh,
v + OFFSET_U,
Vec2::X * c.vel.x * vel_scale,
[1., 0.1, 0.1],
flanges,
);
draw_arrow(
mesh,
v + OFFSET_V,
Vec2::Y * c.vel.y * vel_scale,
[0.01, 0.3, 1.],
flanges,
);
}
}
}

fn draw_grid(mesh: &mut Mesh, grid: &Array2D<GridCell>) {
let color = [0.05; 3];

for y in 0..=grid.height() {
let mut vertex =
|pt: Vec2| mesh.push_vertex(Vertex::new(simspace_to_modelspace(pt), color));
let a = vertex(Vec2::new(0., y as f32));
let b = vertex(Vec2::new(grid.width() as f32, y as f32));
mesh.push_indices(&[a, b]);
}

for x in 0..=grid.width() {
let mut vertex =
|pt: Vec2| mesh.push_vertex(Vertex::new(simspace_to_modelspace(pt), color));
let a = vertex(Vec2::new(x as f32, 0.));
let b = vertex(Vec2::new(x as f32, grid.height() as f32));
mesh.push_indices(&[a, b]);
}
}
*/

/// Display colors and physical behaviour coefficients
#[derive(Clone, Debug)]
pub struct LifeConfig {
    /// Colors of each type
    pub colors: Vec<[f32; 3]>,
    /// Behaviour matrix
    pub behaviours: Array2D<Behaviour>,
}

pub type ParticleType = u8;

#[derive(Clone, Copy, Debug)]
pub struct Behaviour {
    /// Magnitude of the default repulsion force
    pub default_repulse: f32,
    /// Zero point between default repulsion and particle interaction (0 to 1)
    pub inter_threshold: f32,
    /// Interaction peak strength
    pub inter_strength: f32,
    /// Maximum distance of particle interaction (0 to 1)
    pub max_inter_dist: f32,
}

impl Behaviour {
    /// Returns the force on this particle
    ///
    /// Distance is in the range `0.0..=1.0`
    pub fn force(&self, dist: f32) -> f32 {
        if dist < self.inter_threshold {
            let f = dist / self.inter_threshold;
            (1. - f) * -self.default_repulse
        } else if dist > self.max_inter_dist {
            0.0
        } else {
            let x = dist - self.inter_threshold;
            let x = x / (self.max_inter_dist - self.inter_threshold);
            let x = x * 2. - 1.;
            let x = 1. - x.abs();
            x * self.inter_strength
        }
    }
}

/// https://gist.github.com/fairlight1337/4935ae72bcbcc1ba5c72
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let c = v * s; // Chroma
    let h_prime = (h / 60.0) % 6.0;
    let x = c * (1.0 - ((h_prime % 2.0) - 1.0).abs());
    let m = v - c;

    let (mut r, mut g, mut b);

    if (0. ..1.).contains(&h_prime) {
        r = c;
        g = x;
        b = 0.0;
    } else if (1.0..2.0).contains(&h_prime) {
        r = x;
        g = c;
        b = 0.0;
    } else if (2.0..3.0).contains(&h_prime) {
        r = 0.0;
        g = c;
        b = x;
    } else if (3.0..4.0).contains(&h_prime) {
        r = 0.0;
        g = x;
        b = c;
    } else if (4.0..5.0).contains(&h_prime) {
        r = x;
        g = 0.0;
        b = c;
    } else if (5.0..6.0).contains(&h_prime) {
        r = c;
        g = 0.0;
        b = x;
    } else {
        r = 0.0;
        g = 0.0;
        b = 0.0;
    }

    r += m;
    g += m;
    b += m;

    [r, g, b]
}

impl LifeConfig {
    pub fn max_interaction_radius(&self) -> f32 {
        self.behaviours
            .data()
            .iter()
            .map(|b| b.max_inter_dist)
            .max_by(|a, b| a.total_cmp(b))
            .unwrap()
    }

    pub fn min_interaction_radius(&self) -> f32 {
        self.behaviours
            .data()
            .iter()
            .map(|b| b.max_inter_dist)
            .min_by(|a, b| a.total_cmp(b))
            .unwrap()
    }


    pub fn get_behaviour(&self, a: ParticleType, b: ParticleType) -> Behaviour {
        self.behaviours[(a as usize, b as usize)]
    }

    pub fn random_behaviour(std_dev: f32) -> Behaviour {
        let mut rng = rand::thread_rng();
        let mut behav = Behaviour::default();

        let dist = rand_distr::Normal::new(0., std_dev).unwrap();
        behav.inter_strength = dist.sample(&mut rng);
        /*if behav.inter_strength < 0. {
            behav.inter_strength *= 10.;
        }*/
        behav
    }

    pub fn random(rule_count: usize, std_dev: f32) -> Self {
        let mut rng = rand::thread_rng();

        let colors: Vec<[f32; 3]> = (0..rule_count).map(|_| random_color(&mut rng)).collect();
        let behaviours = (0..rule_count.pow(2))
            .map(|_| Self::random_behaviour(std_dev))
            .collect();
        let behaviours = Array2D::from_array(rule_count, behaviours);

        Self { behaviours, colors }
    }
}

impl Default for Behaviour {
    fn default() -> Self {
        Self {
            //default_repulse: 10.,
            max_inter_dist: 4.,
            default_repulse: 20.,
            inter_threshold: 2.,

            inter_strength: 1.,
        }
    }
}

fn random_color(rng: &mut impl Rng) -> [f32; 3] {
    hsv_to_rgb(rng.gen_range(0.0..=360.0), 1., 1.)
}

fn color_to_egui([r, g, b]: [f32; 3]) -> Rgba {
    Rgba::from_rgb(r, g, b)
}

/// Maps sim coordinates to/from egui coordinates
struct CoordinateMapping {
    width: f32,
    height: f32,
    area: Rect,
}

impl CoordinateMapping {
    pub fn new(grid: &Array2D<GridCell>, area: Rect) -> Self {
        Self {
            width: grid.width() as f32 - 1.,
            height: grid.height() as f32 - 1.,
            area,
        }
    }

    pub fn sim_to_egui_vect(&self, pt: glam::Vec2) -> egui::Vec2 {
        egui::Vec2::new(
            (pt.x / self.width) * self.area.width(),
            (-pt.y / self.height) * self.area.height(),
        )
    }

    pub fn sim_to_egui(&self, pt: glam::Vec2) -> egui::Pos2 {
        egui::Pos2::new(
            (pt.x / self.width) * self.area.width(),
            (1. - pt.y / self.height) * self.area.height(),
        )
    }

    pub fn egui_to_sim(&self, pt: egui::Pos2) -> glam::Vec2 {
        glam::Vec2::new(
            (pt.x / self.area.width()) * self.width,
            (1. - pt.y / self.area.height()) * self.height,
        )
    }

    pub fn egui_to_sim_vector(&self, pt: egui::Vec2) -> glam::Vec2 {
        glam::Vec2::new(
            (pt.x / self.area.width()) * self.width,
            (-pt.y / self.area.height()) * self.height,
        )
    }
}

fn move_particles_from_egui(
    particles: &mut [Particle],
    radius: f32,
    coords: &CoordinateMapping,
    _dt: f32,
    response: egui::Response,
) {
    if response.dragged() {
        // pos/frame * (dt/frame)^-1 = pos/dt = velocity
        let vel = coords.egui_to_sim_vector(response.drag_delta());
        let vel = vel * 5.;
        if let Some(pos) = response.interact_pointer_pos() {
            let pos = coords.egui_to_sim(pos - coords.area.left_top().to_vec2());
            push_particles(particles, radius, pos, vel);
        }
    }
}

fn push_particles(particles: &mut [Particle], radius: f32, pos: Vec2, vel: Vec2) {
    let radius_sq = radius.powi(2);
    for part in particles {
        if part.pos.distance_squared(pos) < radius_sq {
            part.vel += vel;
        }
    }
}

/// Inverse of grid_id
fn index_to_pos((i, j): GridPos) -> Vec2 {
    Vec2::new(i as f32, j as f32)
}
