use array2d::{Array2D, GridPos};
use cimvr_common::{
    glam::Vec2,
    render::{Mesh, MeshHandle, Primitive, Render, UploadMesh, Vertex},
    ui::{egui::{DragValue, Slider}, GuiInputMessage, GuiTab},
    Transform,
};
use cimvr_engine_interface::{make_app_state, pcg::Pcg, pkg_namespace, prelude::*};
use query_accel::QueryAccelerator;
use rand::prelude::*;

mod array2d;
mod query_accel;

struct ClientState {
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
    pic_flip_ratio: f32,

    well: bool,
    source: bool,

    ui: GuiTab,
}

make_app_state!(ClientState, DummyUserState);

const POINTS_RDR: MeshHandle = MeshHandle::new(pkg_namespace!("Points"));
const LINES_RDR: MeshHandle = MeshHandle::new(pkg_namespace!("Lines"));

impl UserState for ClientState {
    fn new(io: &mut EngineIo, sched: &mut EngineSchedule<Self>) -> Self {
        io.create_entity()
            .add_component(Transform::default())
            .add_component(Render::new(POINTS_RDR).primitive(Primitive::Points))
            .build();

        io.create_entity()
            .add_component(Transform::default())
            .add_component(Render::new(LINES_RDR).primitive(Primitive::Lines))
            .build();

        sched.add_system(Self::update).build();

        sched
            .add_system(Self::update_gui)
            .subscribe::<GuiInputMessage>()
            .build();

        let width = 100;
        let height = 100;
        let n_particles = 10_000;
        let particle_radius = 0.36;
        let sim = Sim::new(width, height, n_particles, particle_radius);

        Self {
            pic_flip_ratio: 0.95,
            calc_rest_density_from_radius: true,
            single_step: false,
            dt: 0.04,
            solver_iters: 100,
            stiffness: 3.,
            gravity: 9.8,
            sim,
            ui: GuiTab::new(io, "PIC Fluids"),
            width,
            height,
            n_particles,
            solver: IncompressibilitySolver::GaussSeidel,
            well: false,
            source: false,
            show_arrows: false,
            pause: false,
            grid_vel_scale: 0.05,
            show_grid: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IncompressibilitySolver {
    Jacobi,
    GaussSeidel,
}

impl ClientState {
    fn update(&mut self, io: &mut EngineIo, _query: &mut QueryResult) {
        // Update
        if !self.pause || self.single_step {
            if self.source {
                let pos = Vec2::new(10., 90.);
                //let vel = Vec2::new(0., -20.);
                let vel = Vec2::ZERO;
                let deriv = [Vec2::ZERO; 2];
                self.sim.particles.push(Particle { pos, vel, deriv });
            }

            if self.well {
                for part in &mut self.sim.particles {
                    if part.pos.x < 20. {
                        part.vel += self.dt * 9.;
                    }
                }
            }

            self.sim.step(
                self.dt,
                self.solver_iters,
                self.stiffness,
                self.gravity,
                self.pic_flip_ratio,
                self.solver,
            );

            self.single_step = false;
        }

        // Display
        io.send(&UploadMesh {
            mesh: particles_mesh(&self.sim.particles),
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
    }

    fn update_gui(&mut self, io: &mut EngineIo, _query: &mut QueryResult) {
        self.ui.show(io, |ui| {
            ui.add(Slider::new(&mut self.pic_flip_ratio, 0.0..=1.0).text("PIC - FLIP"));
            ui.add(DragValue::new(&mut self.stiffness).prefix("Stiffness: "));
            ui.add(
                DragValue::new(&mut self.dt)
                    .prefix("Δt (time step): ")
                    .speed(1e-3),
            );
            ui.add(DragValue::new(&mut self.solver_iters).prefix("Solver iterations: "));
            ui.add(
                DragValue::new(&mut self.gravity)
                    .prefix("Gravity: ")
                    .speed(1e-2),
            );
            ui.add(
                DragValue::new(&mut self.sim.particle_radius)
                    .prefix("Particle radius: ")
                    .speed(1e-2)
                    .clamp_range(1e-2..=5.0),
            );
            ui.horizontal(|ui| {
                ui.add(
                    DragValue::new(&mut self.sim.rest_density)
                        .prefix("Rest density: ")
                        .speed(1e-2),
                );
                ui.checkbox(&mut self.calc_rest_density_from_radius, "From radius");
                if self.calc_rest_density_from_radius {
                    self.sim.rest_density = calc_rest_density(self.sim.particle_radius);
                }
            });
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.pause, "Pause");
                self.single_step |= ui.button("Step").clicked();
            });
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

            ui.separator();
            ui.strong("Incompressibility Solver");
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

            ui.separator();
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
            ui.add(
                DragValue::new(&mut self.n_particles)
                    .prefix("# of particles: ")
                    .clamp_range(1..=usize::MAX)
                    .speed(4),
            );

            if ui.button("Reset").clicked() {
                self.sim = Sim::new(
                    self.width,
                    self.height,
                    self.n_particles,
                    self.sim.particle_radius,
                );
            }

            ui.separator();
            ui.checkbox(&mut self.source, "Particle source");
            ui.checkbox(&mut self.well, "Particle well");
        });
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

fn particles_mesh(particles: &[Particle]) -> Mesh {
    Mesh {
        vertices: particles
            .iter()
            .map(|p| Vertex::new(simspace_to_modelspace(p.pos), [1.; 3]))
            .collect(),
        indices: (0..particles.len() as u32).collect(),
    }
}

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
    /// Velocity derivatives
    deriv: [Vec2; 2],
}

fn calc_rest_density(particle_radius: f32) -> f32 {
    // Assume hexagonal packing
    let packing_density = std::f32::consts::PI / 2. / 3_f32.sqrt();
    let particle_area = std::f32::consts::PI * particle_radius.powi(2);
    packing_density / particle_area
}

impl Sim {
    pub fn new(width: usize, height: usize, n_particles: usize, particle_radius: f32) -> Self {
        // Uniformly placed, random particles
        let mut rng = rng();
        let particles = (0..n_particles)
            .map(|_| {
                let pos = Vec2::new(
                    rng.gen_range(1.0..=(width - 2) as f32),
                    rng.gen_range(1.0..=(height - 2) as f32),
                );
                Particle {
                    pos,
                    vel: Vec2::ZERO,
                    deriv: [Vec2::ZERO; 2],
                }
            })
            .collect();

        // Assuming perfect hexagonal packing,
        // Packing efficiency * (1 / particle area) = particles / area

        Sim {
            particles,
            grid: Array2D::new(width, height),
            rest_density: calc_rest_density(particle_radius),
            particle_radius,
            over_relax: 1.5,
        }
    }

    pub fn step(
        &mut self,
        dt: f32,
        solver_iters: usize,
        stiffness: f32,
        gravity: f32,
        pic_flip_ratio: f32,
        solver: IncompressibilitySolver,
    ) {
        // Step particles
        apply_global_force(&mut self.particles, Vec2::new(0., -gravity), dt);
        step_particles(&mut self.particles, dt);
        enforce_particle_radius(&mut self.particles, self.particle_radius);
        enforce_particle_pos(&mut self.particles, &self.grid);

        // Step grid
        particles_to_grid(&self.particles, &mut self.grid);
        let solver_fn = match solver {
            IncompressibilitySolver::Jacobi => solve_incompressibility_jacobi,
            IncompressibilitySolver::GaussSeidel => solve_incompressibility_gauss_seidel,
        };

        //let old_vel = self.grid.clone();
        solver_fn(
            &mut self.grid,
            solver_iters,
            self.rest_density,
            self.over_relax,
            stiffness,
        );

        grid_to_particles(&mut self.particles, &self.grid);
    }
}

fn rng() -> SmallRng {
    let u = ((Pcg::new().gen_u32() as u64) << 32) | Pcg::new().gen_u32() as u64;
    SmallRng::seed_from_u64(u)
}

/// Move particles forwards in time by `dt`, assuming unit mass for all particles.
fn step_particles(particles: &mut [Particle], dt: f32) {
    for part in particles {
        part.pos += part.vel * dt;
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
fn particles_to_grid(particles: &[Particle], grid: &mut Array2D<GridCell>) {
    // Clear the grid
    grid.data_mut()
        .iter_mut()
        .for_each(|c| *c = GridCell::default());


    // Accumulate velocity on grid
    // Here we abuse the pressure of each grid cell to by mass correctly
    for part in particles {
        let u_pos = part.pos - OFFSET_U;
        scatter(u_pos, grid, |c, n, w| c.vel.x += w * (part.vel.x + (index_to_pos(n) - u_pos).dot(part.deriv[0])));
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
        scatter(v_pos, grid, |c, n, w| c.vel.y += w * (part.vel.y + (index_to_pos(n) - v_pos).dot(part.deriv[1])));
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

/// Inverse of grid_id
fn index_to_pos((i, j): GridPos) -> Vec2 {
    Vec2::new(i as f32, j as f32)
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

        // Interpolate grid vectors
        part.deriv[0] = gather_vector(u_pos, |p| grid[p].vel.x * (index_to_pos(p) - u_pos).normalize());
        part.deriv[1] = gather_vector(v_pos, |p| grid[p].vel.y * (index_to_pos(p) - v_pos).normalize());
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

    //let mut rng = rng();

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
