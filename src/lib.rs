use array2d::{Array2D, GridPos};
use cimvr_common::{
    glam::Vec2,
    render::{Mesh, MeshHandle, Primitive, Render, UploadMesh, Vertex},
    Transform,
};
use cimvr_engine_interface::{dbg, make_app_state, pcg::Pcg, pkg_namespace, prelude::*};
use rand::prelude::*;

mod array2d;

struct ClientState {
    // Sim state
    sim: Sim,

    // Settings
    dt: f32,
    solver_iters: usize,
    stiffness: f32,
    gravity: f32,
}

make_app_state!(ClientState, DummyUserState);

const POINTS_RDR: MeshHandle = MeshHandle::new(pkg_namespace!("Points"));

impl UserState for ClientState {
    fn new(io: &mut EngineIo, sched: &mut EngineSchedule<Self>) -> Self {
        io.create_entity()
            .add_component(Transform::default())
            .add_component(Render::new(POINTS_RDR).primitive(Primitive::Points))
            .build();

        sched.add_system(Self::update).build();

        let sim = Sim::new(100, 100, 2_000);

        Self {
            dt: 0.001,
            solver_iters: 50,
            stiffness: 0.0,
            gravity: 10.,
            sim,
        }
    }
}

impl ClientState {
    fn update(&mut self, io: &mut EngineIo, _query: &mut QueryResult) {
        self.sim
            .step(self.dt, self.solver_iters, self.stiffness, self.gravity);

        io.send(&UploadMesh {
            mesh: particles_mesh(&self.sim.particles),
            id: POINTS_RDR,
        });
    }
}

fn particles_mesh(particles: &[Particle]) -> Mesh {
    const DOWNSCALE: f32 = 100.;
    Mesh {
        vertices: particles
            .iter()
            .map(|p| {
                Vertex::new(
                    [
                        (p.pos.x / DOWNSCALE) * 2. - 1.,
                        0.,
                        (p.pos.y / DOWNSCALE) * 2. - 1.,
                    ],
                    [1.; 3],
                )
            })
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
}

impl Sim {
    pub fn new(width: usize, height: usize, n_particles: usize) -> Self {
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
                }
            })
            .collect();

        let rest_density = n_particles as f32 / ((width - 2) * (height - 2)) as f32;

        Sim {
            particles,
            grid: Array2D::new(width, height),
            rest_density,
        }
    }

    pub fn step(&mut self, dt: f32, solver_iters: usize, stiffness: f32, gravity: f32) {
        // Step particles
        apply_global_force(&mut self.particles, Vec2::new(0., -gravity), dt);
        step_particles(&mut self.particles, dt);
        enforce_particle_pos(&mut self.particles, &self.grid);

        // Step grid
        particles_to_grid(&self.particles, &mut self.grid);
        enforce_grid_boundary(&mut self.grid);
        solve_incompressibility(
            &mut self.grid,
            solver_iters,
            self.rest_density,
            stiffness,
            1.9,
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
    // Here we abuse the pressure of each grid cell to divide correctly
    for part in particles {
        scatter(part.pos - OFFSET_U, grid, |c, w| c.vel.x += w * part.vel.x);
        scatter(part.pos - OFFSET_U, grid, |c, w| c.pressure += w);
    }
    grid.data_mut().iter_mut().for_each(|c| {
        if c.pressure != 0.0 {
            c.vel.x /= c.pressure;
        }
    });
    grid.data_mut().iter_mut().for_each(|c| c.pressure = 0.);

    // And then we do again for u
    for part in particles {
        scatter(part.pos - OFFSET_V, grid, |c, w| c.vel.y += w * part.vel.y);
        scatter(part.pos - OFFSET_V, grid, |c, w| c.pressure += w);
    }
    grid.data_mut().iter_mut().for_each(|c| {
        if c.pressure != 0.0 {
            c.vel.y /= c.pressure;
        }
    });
    grid.data_mut().iter_mut().for_each(|c| c.pressure = 0.);

    // Now we actually set the pressure
    for part in particles {
        scatter(part.pos, grid, |c, w| c.pressure += w);
    }
}

/// Returns the weights for each grid
/// corner in the order [tl, tr, bl, br]
///
/// Panics if coordinates are inverted
fn weights(pos: Vec2) -> [f32; 4] {
    let left = pos.x.fract();
    let right = 1. - left;
    let top = pos.y.fract();
    let bottom = 1. - top;

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

/// Performs a weighted accumulation on the grid field chosen by f
fn scatter<T>(pos: Vec2, grid: &mut Array2D<T>, mut f: impl FnMut(&mut T, f32)) {
    let weights = weights(pos);
    let neighbors = grid_neighborhood(grid_tl(pos));

    for (w, n) in weights.into_iter().zip(neighbors) {
        f(&mut grid[n], w);
    }
}

/// Solve incompressibility on the grid cells, includinge contribution from presssure
fn solve_incompressibility(
    grid: &mut Array2D<GridCell>,
    iterations: usize,
    rest_density: f32,
    overrelaxation: f32,
    stiffness: f32,
) {
    // TODO: Use Jacobi method instead!
    for _ in 0..iterations {
        for i in 1..grid.width() - 1 {
            for j in 1..grid.height() - 1 {
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
}

fn grid_to_particles(particles: &mut [Particle], grid: &Array2D<GridCell>) {
    for part in particles {
        // Interpolate velocity onto particles
        let vel_x = gather(part.pos - OFFSET_U, grid, |c| c.vel.x);
        let vel_y = gather(part.pos - OFFSET_V, grid, |c| c.vel.y);
        part.vel = Vec2::new(vel_x, vel_y);
    }
}

fn enforce_particle_pos(particles: &mut [Particle], grid: &Array2D<GridCell>) {
    for part in particles {
        // Ensure particles are within the grid
        part.pos.x = part.pos.x.clamp(1.0, (grid.width() - 2) as f32);
        part.pos.y = part.pos.y.clamp(1.0, (grid.height() - 2) as f32);
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
