use crate::array2d::Array2D;

use eframe::egui::{DragValue, Grid, Rgba, RichText, ScrollArea, Slider, Ui};
use egui::os::OperatingSystem;
use egui::SidePanel;
use egui::{CentralPanel, Frame, Rect, Sense};
use glam::Vec2;
use vorpal_widgets::node_editor::NodeGraphWidget;


use crate::sim::*;

/// We derive Deserialize/Serialize so we can persist app state on shutdown.
pub struct TemplateApp {
    // Sim state
    sim: Sim,

    nodes: NodeGraphWidget,
    node_cfg: NodeInteractionCfg,
    particle_mode: ParticleBehaviourMode,

    // Settings
    tweak: SimTweak,
    width: usize,
    height: usize,
    calc_rest_density_from_radius: bool,
    set_inter_dist_to_radius: bool,
    //show_arrows: bool,
    //show_grid: bool,
    //grid_vel_scale: f32,
    pause: bool,
    single_step: bool,
    n_colors: usize,
    random_std_dev: f32,

    well: bool,
    source_color_idx: ParticleType,
    source_rate: usize,
    //mult: f32,
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
        let particle_radius = 0.28;
        let random_std_dev = 5.;

        let n_colors = 3;
        let life = LifeConfig::random(n_colors, random_std_dev);
        let sim = Sim::new(width, height, n_particles, particle_radius, life);

        let nodes = NodeGraphWidget::new(nodegraph_fn_inputs());
        let node_cfg = NodeInteractionCfg::default();

        Self {
            tweak: SimTweak::default(),
            node_cfg,
            particle_mode: ParticleBehaviourMode::NodeGraph,
            nodes,
            advanced: false,
            n_colors,
            source_rate: 0,
            calc_rest_density_from_radius: true,
            single_step: false,
            sim,
            width,
            height,
            well: false,
            source_color_idx: 0,
            random_std_dev,
            //show_arrows: false,
            pause: false,
            //grid_vel_scale: 0.05,
            //show_grid: false,
            show_settings_only: false,
            set_inter_dist_to_radius: true,
            //mult: 1.0,
        }
    }
}

fn is_mobile(ctx: &egui::Context) -> bool {
    matches!(ctx.os(), OperatingSystem::Android | OperatingSystem::IOS)
}

impl eframe::App for TemplateApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        #[cfg(not(target_arch = "wasm32"))]
        puffin::GlobalProfiler::lock().new_frame();

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

            if self.particle_mode == ParticleBehaviourMode::NodeGraph {
                SidePanel::right("NodeGraph").show(ctx, |ui| {
                    self.nodes.show(ui);
                });
            }

            CentralPanel::default().show(ctx, |ui| {
                Frame::canvas(ui.style()).show(ui, |ui| self.sim_widget(ui))
            });
        }
    }
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
                        part.vel += self.tweak.dt * 9.;
                    }
                }
            }

            let node = vorpal_widgets::vorpal_core::highlevel::convert_node(
                self.nodes.extract_output_node(),
            );

            self.sim.step(&self.tweak, &self.node_cfg, &node);

            self.single_step = false;
        }
    }

    fn sim_widget(&mut self, ui: &mut Ui) {
        let (rect, response) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());

        let coords = CoordinateMapping::new(&self.sim.grid, rect);

        // Move particles
        move_particles_from_egui(
            &mut self.sim.particles,
            4.,
            &coords,
            self.tweak.dt,
            response,
        );

        // Step particles
        if !self.pause || self.single_step {
            self.update();
            self.single_step = false;
        }

        // Draw particles
        let painter = ui.painter_at(rect);
        let radius = coords
            .sim_to_egui_vect(Vec2::splat(self.sim.particle_radius))
            .length()
            / 2_f32.sqrt();

        for part in &self.sim.particles {
            let color = self.sim.life.colors[part.color as usize];
            painter.circle_filled(
                coords.sim_to_egui(part.pos) + rect.left_top().to_vec2(),
                radius,
                color_to_egui(color),
            );
        }
    }

    fn settings_gui(&mut self, ui: &mut Ui) {
        let mut reset = false;
        ui.separator();
        ui.strong("Simulation state");

        let mut n_particles = self.sim.particles.len();
        if ui
            .add(
                DragValue::new(&mut n_particles)
                    .prefix("# of particles: ")
                    .clamp_range(1..=usize::MAX)
                    .speed(4),
            )
            .changed()
        {
            let mut rng = rand::thread_rng();
            self.sim.particles.resize_with(n_particles, || {
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
                        new_behav_array[(i, j)] = LifeConfig::random_behaviour(self.random_std_dev);
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
            DragValue::new(&mut self.tweak.dt)
                .prefix("Δt (time step): ")
                .speed(1e-4),
        );
        ui.horizontal(|ui| {
            ui.add(
                DragValue::new(&mut self.tweak.gravity)
                    .prefix("Gravity: ")
                    .speed(1e-2),
            );
            if ui.button("Zero-G").clicked() {
                self.tweak.gravity = 0.;
            }
        });
        if self.advanced {
            ui.add(Slider::new(&mut self.sim.damping, 0.0..=1.0).text("Damping"));
            ui.checkbox(
                &mut self.tweak.enable_grid_transfer,
                "Grid transfer (required for incompressibility solver!)",
            );
            ui.add(Slider::new(&mut self.tweak.pic_apic_ratio, 0.0..=1.0).text("PIC - APIC"));
        }

        ui.separator();
        ui.horizontal(|ui| {
            ui.strong("Particle collisions");
        });
        ui.add(
            DragValue::new(&mut self.sim.particle_radius)
                .prefix("Particle radius: ")
                .speed(1e-2)
                .clamp_range(1e-2..=5.0),
        );
        if self.advanced {
            ui.checkbox(
                &mut self.tweak.enable_particle_collisions,
                "Hard collisions",
            );
            ui.horizontal(|ui| {
                ui.add(
                    DragValue::new(&mut self.sim.rest_density)
                        .prefix("Rest density: ")
                        .speed(1e-2),
                );
                ui.checkbox(
                    &mut self.calc_rest_density_from_radius,
                    "From radius (assumes optimal packing)",
                );
            });
        }

        if self.calc_rest_density_from_radius {
            self.sim.rest_density = calc_rest_density(self.sim.particle_radius);
        }

        if self.advanced && self.tweak.enable_grid_transfer {
            ui.separator();
            ui.horizontal(|ui| {
                ui.strong("Incompressibility Solver");
                ui.checkbox(&mut self.tweak.enable_incompress, "");
            });
            ui.add(DragValue::new(&mut self.tweak.solver_iters).prefix("Solver iterations: "));
            ui.add(
                DragValue::new(&mut self.sim.over_relax)
                    .prefix("Over-relaxation: ")
                    .speed(1e-2)
                    .clamp_range(0.0..=1.95),
            );
            ui.horizontal(|ui| {
                ui.selectable_value(
                    &mut self.tweak.solver,
                    IncompressibilitySolver::Jacobi,
                    "Jacobi",
                );
                ui.selectable_value(
                    &mut self.tweak.solver,
                    IncompressibilitySolver::GaussSeidel,
                    "Gauss Seidel",
                );
            });
            ui.add(
                DragValue::new(&mut self.tweak.stiffness)
                    .prefix("Density compensation stiffness: ")
                    .speed(1e-2),
            );
        }

        ui.separator();
        ui.strong("Particle source");
        ui.add(
            DragValue::new(&mut self.source_rate)
                .prefix("Particle inflow rate: ")
                .speed(1e-1),
        );
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
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.particle_mode, ParticleBehaviourMode::Off, "Off");
            ui.selectable_value(
                &mut self.particle_mode,
                ParticleBehaviourMode::NodeGraph,
                "Node graph",
            );
            ui.selectable_value(
                &mut self.particle_mode,
                ParticleBehaviourMode::ParticleLife,
                "Particle life",
            );
        });

        if self.particle_mode == ParticleBehaviourMode::NodeGraph {
            ui.add(
                DragValue::new(&mut self.node_cfg.neighbor_radius)
                    .clamp_range(1e-2..=20.0)
                    .speed(1e-2)
                    .prefix("Neighbor_radius: "),
            );
        }

        if self.particle_mode == ParticleBehaviourMode::ParticleLife {
            let mut behav_cfg = self.sim.life.behaviours[(0, 0)];
            if self.advanced {
                ui.add(
                    DragValue::new(&mut behav_cfg.max_inter_dist)
                        .clamp_range(0.0..=20.0)
                        .speed(1e-2)
                        .prefix("Max interaction dist: "),
                );
                ui.add(
                    DragValue::new(&mut behav_cfg.default_repulse)
                        .speed(1e-2)
                        .prefix("Default repulse: "),
                );
                ui.horizontal(|ui| {
                    ui.add(
                        DragValue::new(&mut behav_cfg.inter_threshold)
                            .clamp_range(0.0..=20.0)
                            .speed(1e-2)
                            .prefix("Interaction threshold: "),
                    );
                    ui.checkbox(&mut self.set_inter_dist_to_radius, "From radius");
                });
            }
            if self.set_inter_dist_to_radius {
                behav_cfg.inter_threshold = self.sim.particle_radius * 2.;
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
                    self.sim.life = LifeConfig::random(self.n_colors, self.random_std_dev);
                    reset = true;
                }
                ui.add(
                    DragValue::new(&mut self.random_std_dev)
                        .prefix("std. dev: ")
                        .speed(1e-2)
                        .clamp_range(0.0..=f32::MAX),
                )
            });
            if ui.button("Symmetric forces").clicked() {
                let n = self.sim.life.colors.len();
                for i in 0..n {
                    for j in 0..i {
                        self.sim.life.behaviours[(i, j)] = self.sim.life.behaviours[(j, i)]
                    }
                }
            }
        }

        /*
        if ui.button("Lifeless").clicked() {
            self.sim
                .life
                .behaviours
                .data_mut()
                .iter_mut()
                .for_each(|b| b.inter_strength = 0.);
        }
        */

        /*
        if self.advanced {
            ui.add(DragValue::new(&mut self.mult).speed(1e-2));
        }
        */

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
                self.sim.particles.len(),
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

/*
const SIM_TO_MODEL_DOWNSCALE: f32 = 100.;
fn simspace_to_modelspace(pos: Vec2) -> [f32; 3] {
    [
        (pos.x / SIM_TO_MODEL_DOWNSCALE) * 2. - 1.,
        0.,
        (pos.y / SIM_TO_MODEL_DOWNSCALE) * 2. - 1.,
    ]
}
*/

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

fn color_to_egui([r, g, b]: [f32; 3]) -> Rgba {
    Rgba::from_rgb(r, g, b)
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
