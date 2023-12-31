use crate::array2d::Array2D;

use eframe::egui::{DragValue, Grid, Rgba, RichText, ScrollArea, Slider, Ui};
use egui::os::OperatingSystem;
use egui::{CentralPanel, Frame, Rect, Sense};
use egui::{Color32, Painter, SidePanel};
use glam::Vec2;

use crate::sim::*;

/// We derive Deserialize/Serialize so we can persist app state on shutdown.
pub struct TemplateApp {
    // Sim state
    sim: Sim,

    // Settings
    pause: bool,
    single_step: bool,
    advanced: bool,
    show_settings_only: bool,

    n_protons: usize,
    n_electrons: usize,
    tweak: SimTweaks,
}

impl TemplateApp {
    /// Called once before the first frame.
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let n_protons = 25;
        let n_electrons = 50;

        let sim = Sim::new(n_protons, n_electrons);

        Self {
            n_protons,
            n_electrons,
            tweak: SimTweaks::default(),
            sim,
            pause: false,
            advanced: false,
            single_step: false,
            show_settings_only: is_mobile(&cc.egui_ctx),
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
            self.sim.step(&self.tweak);
            self.single_step = false;
        }
    }

    fn sim_widget(&mut self, ui: &mut Ui) {
        let (rect, response) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());

        let coords = CoordinateMapping::new(100, 100, rect);

        // Move particles
        //move_particles_from_egui(&mut self.sim.particles, 4., &coords, self.dt, response);

        // Step particles
        if !self.pause || self.single_step {
            self.update();
            self.single_step = false;
        }

        // Draw particles
        let painter = ui.painter_at(rect);
        draw_sim(&self.sim, &coords, &rect, &painter);
    }

    fn settings_gui(&mut self, ui: &mut Ui) {
        let mut reset = false;
        ui.separator();
        ui.strong("Simulation state");

        ui.horizontal(|ui| {
            ui.checkbox(&mut self.pause, "Pause");
            self.single_step |= ui.button("Step").clicked();
        });
        if ui.button("Reset").clicked() {
            reset = true;
        }

        ui.separator();

        ui.add(
            DragValue::new(&mut self.n_protons)
                .prefix("# of protons: ")
                .speed(1e-1),
        );
        ui.add(
            DragValue::new(&mut self.n_electrons)
                .prefix("# of electrons: ")
                .speed(1e-1),
        );

        ui.label("Protons");
        ui.strong("Kinematics");
        ui.add(
            DragValue::new(&mut self.tweak.proton_dt)
                .prefix("Δt (time step): ")
                .speed(1e-4),
        );
        ui.add(
            DragValue::new(&mut self.tweak.proton_electron_smooth)
                .prefix("p+ to e- smooth: ")
                .speed(1e-3),
        );
        ui.add(
            DragValue::new(&mut self.tweak.proton_proton_smooth)
                .prefix("p+ to p+ smooth: ")
                .speed(1e-3),
        );

        ui.label("Electrons");
        ui.add(
            DragValue::new(&mut self.tweak.electron_steps)
                .prefix("e- steps: ")
                .speed(1e-1),
        );
        ui.add(
            DragValue::new(&mut self.tweak.electron_sigma)
                .prefix("e- sigma: ")
                .speed(1e-3),
        );
        ui.add(
            DragValue::new(&mut self.tweak.electron_temperature)
                .prefix("e- temp: ")
                .speed(1e-5),
        );
        ui.add(
            DragValue::new(&mut self.tweak.electron_electron_smooth)
                .prefix("e- to e- smooth: ")
                .speed(1e-3),
        );
        ui.add(
            DragValue::new(&mut self.tweak.electron_proton_smooth)
                .prefix("e- to p+ smooth: ")
                .speed(1e-3),
        );

        if reset {
            self.sim = Sim::new(self.n_protons, self.n_electrons);
        }

        ui.checkbox(&mut self.advanced, "Advanced settings");
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
    pub fn new(width: usize, height: usize, area: Rect) -> Self {
        Self {
            width: width as f32 - 1.,
            height: height as f32 - 1.,
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

/*
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
*/

fn color_to_egui([r, g, b]: [f32; 3]) -> Rgba {
    Rgba::from_rgb(r, g, b)
}

fn draw_sim(sim: &Sim, coords: &CoordinateMapping, rect: &Rect, painter: &Painter) {
    for part in &sim.protons {
        painter.circle_filled(
            coords.sim_to_egui(part.pos) + rect.left_top().to_vec2(),
            3.0,
            Color32::RED,
        );
    }

    for part in &sim.electrons {
        painter.circle_filled(
            coords.sim_to_egui(part.pos) + rect.left_top().to_vec2(),
            1.0,
            Color32::YELLOW,
        );
    }
}
