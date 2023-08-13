use array2d::Array2D;
use cimvr_common::{
    glam::Vec2,
    render::{Mesh, MeshHandle, Primitive, Render, UploadMesh, Vertex},
    Transform,
};
use cimvr_engine_interface::{make_app_state, pkg_namespace, prelude::*, pcg::Pcg};
use rand::prelude::*;
use zwohash::HashMap;

mod array2d;

struct ClientState;

make_app_state!(ClientState, DummyUserState);

const POINTS_RDR: MeshHandle = MeshHandle::new(pkg_namespace!("Points"));

impl UserState for ClientState {
    fn new(io: &mut EngineIo, sched: &mut EngineSchedule<Self>) -> Self {
        io.send(&UploadMesh {
            mesh: cube(),
            id: POINTS_RDR,
        });

        io.create_entity()
            .add_component(Transform::default())
            .add_component(Render::new(POINTS_RDR).primitive(Primitive::Points))
            .build();

        sched.add_system(Self::update).build();

        Self
    }
}

impl ClientState {
    fn update(&mut self, io: &mut EngineIo, _query: &mut QueryResult) {}
}

fn cube() -> Mesh {
    let size = 0.25;

    let vertices = vec![
        Vertex::new([-size, -size, -size], [0.0, 1.0, 1.0]),
        Vertex::new([size, -size, -size], [1.0, 0.0, 1.0]),
        Vertex::new([size, size, -size], [1.0, 1.0, 0.0]),
        Vertex::new([-size, size, -size], [0.0, 1.0, 1.0]),
        Vertex::new([-size, -size, size], [1.0, 0.0, 1.0]),
        Vertex::new([size, -size, size], [1.0, 1.0, 0.0]),
        Vertex::new([size, size, size], [0.0, 1.0, 1.0]),
        Vertex::new([-size, size, size], [1.0, 0.0, 1.0]),
    ];

    let indices = vec![
        3, 1, 0, 2, 1, 3, 2, 5, 1, 6, 5, 2, 6, 4, 5, 7, 4, 6, 7, 0, 4, 3, 0, 7, 7, 2, 3, 6, 2, 7,
        0, 5, 4, 1, 5, 0,
    ];

    Mesh { vertices, indices }
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
    /// Velocity of the faces represented by this cell
    vel: Vec2,
    /// Pressure inside this cell
    p: f32,
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

        let rest_density = n_particles as f32 / ((width-2) * (height-2)) as f32;

        Sim {
            particles,
            grid: Array2D::new(width, height),
            rest_density,
        }
    }

    pub fn step(&mut self, dt: f32, solver_iters: usize, stiffness: f32) {
        step_particles(&mut self.particles, dt);
        particles_to_grid(&self.particles, &mut self.grid);
        solve_incompressibility(&mut self.grid, solver_iters, self.rest_density, stiffness);
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

fn particles_to_grid(
    particles: &[Particle],
    grid: &mut Array2D<GridCell>,
) {
}

fn solve_incompressibility(
    grid: &mut Array2D<GridCell>,
    iterations: usize,
    rest_density: f32,
    stiffness: f32,
) {
}

fn grid_to_particles(
    particles: &mut [Particle],
    grid: &Array2D<GridCell>,
) {
}


