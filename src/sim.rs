use crate::array2d::{Array2D, GridPos};
use crate::query_accel::QueryAccelerator;

use glam::Vec2;
use rand::prelude::*;

#[derive(Clone)]
pub struct Sim {
    /// Particles
    pub particles: Vec<Particle>,
    /// Cell wall velocity, staggered grid
    pub grid: Array2D<GridCell>,
    /// Rest density, in particles/unit^2
    pub rest_density: f32,
    pub particle_radius: f32,
    pub over_relax: f32,
    pub life: LifeConfig,
    pub damping: f32,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct GridCell {
    /// Flow rate through the top and left faces of this cell
    pub vel: Vec2,
    /// Pressure inside this cell
    pub pressure: f32,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct Particle {
    /// Position
    pub pos: Vec2,
    /// Velocity
    pub vel: Vec2,
    /// Particle type
    pub color: ParticleType,
    /// Velocity derivatives
    pub deriv: [Vec2; 2],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IncompressibilitySolver {
    Jacobi,
    GaussSeidel,
}

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

pub fn calc_rest_density(particle_radius: f32) -> f32 {
    // Assume hexagonal packing
    let packing_density = std::f32::consts::PI / 2. / 3_f32.sqrt();
    let particle_area = std::f32::consts::PI * particle_radius.powi(2);

    // A guess for particle life
    packing_density / particle_area
}

pub fn random_particle(
    rng: &mut impl Rng,
    width: usize,
    height: usize,
    life: &LifeConfig,
) -> Particle {
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
            over_relax: 1.5,
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
        enable_particle_life: bool,
    ) {
        //puffin::profile_scope!("Complete Step");
        // Step particles
        apply_global_force(&mut self.particles, Vec2::new(0., -gravity), dt);
        if enable_particle_life {
            particle_interactions(&mut self.particles, &mut self.life, dt);
        }
        step_particles(&mut self.particles, dt, self.damping);
        if enable_particle_collisions {
            enforce_particle_radius(&mut self.particles, self.particle_radius);
        }
        enforce_particle_pos(&mut self.particles, &self.grid);

        // Step grid
        if enable_grid_transfer {
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

            grid_to_particles(&mut self.particles, &self.grid);
        }
    }
}

/// Move particles forwards in time by `dt`, assuming unit mass for all particles.
fn step_particles(particles: &mut [Particle], dt: f32, damping: f32) {
    puffin::profile_scope!("Move particles");
    for part in particles {
        part.vel *= 1. - damping;
        part.pos += part.vel * dt;
    }
}

/// Apply a force to all particles, e.g. gravity
fn apply_global_force(particles: &mut [Particle], g: Vec2, dt: f32) {
    puffin::profile_scope!("Applying global force");
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
    puffin::profile_scope!("Transfer particles to grid");
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
    puffin::profile_scope!("Solve incompressibility (Jacobi)");
    let mut tmp = grid.clone();

    let mut grid = grid.clone();
    grid.data_mut().iter_mut().for_each(|cell| cell.vel *= f32::from(cell.pressure > 0.));
    let grid = &mut grid;

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
    puffin::profile_scope!("Solve incompressibility (Gauss-Seidel)");
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
    puffin::profile_scope!("Grid transfer to particles");
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
    puffin::profile_scope!("Keep particles in boundaries");
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
    puffin::profile_scope!("Push particles apart");
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

fn particle_interactions(particles: &mut [Particle], cfg: &LifeConfig, dt: f32) {
    puffin::profile_scope!("Particle interactions");
    let points: Vec<Vec2> = particles.iter().map(|p| p.pos).collect();
    let accel = QueryAccelerator::new(&points, cfg.max_interaction_radius());
    //accel.stats("Life");

    for i in 0..particles.len() {
        for neighbor in accel.query_neighbors_fast(i, points[i]) {
            let a = points[i];
            let b = points[neighbor];

            // The vector pointing from a to b
            let diff = b - a;

            // Distance is capped
            let dist = diff.length();
            if dist > 0. {
                // Accelerate towards b
                let normal = diff.normalize();
                let behav = cfg.get_behaviour(particles[i].color, particles[neighbor].color);
                let accel = normal * behav.force(dist);

                particles[i].vel += accel * dt;
            }
        }
    }
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

    pub fn get_behaviour(&self, a: ParticleType, b: ParticleType) -> Behaviour {
        self.behaviours[(a as usize, b as usize)]
    }

    pub fn random_behaviour(random_std_dev: f32) -> Behaviour {
        let mut rng = rand::thread_rng();
        let mut behav = Behaviour::default();
        behav.inter_strength = rand_distr::Normal::new(0., random_std_dev)
            .map(|distr| distr.sample(&mut rng))
            .unwrap_or(0.);
        if behav.inter_strength < 0. {
            behav.inter_strength *= 10.;
        }
        behav
    }

    pub fn random(rule_count: usize, random_std_dev: f32) -> Self {
        let mut rng = rand::thread_rng();

        let colors: Vec<[f32; 3]> = (0..rule_count).map(|_| random_color(&mut rng)).collect();
        let behaviours = (0..rule_count.pow(2))
            .map(|_| Self::random_behaviour(random_std_dev))
            .collect();
        let behaviours = Array2D::from_array(rule_count, behaviours);

        Self { behaviours, colors }
    }
}

impl Default for Behaviour {
    fn default() -> Self {
        Self {
            //default_repulse: 10.,
            default_repulse: 400.,
            inter_threshold: 0.75,
            inter_strength: 1.,
            max_inter_dist: 2.,
        }
    }
}

pub fn random_color(rng: &mut impl Rng) -> [f32; 3] {
    hsv_to_rgb(rng.gen_range(0.0..=360.0), 1., 1.)
}

/// Inverse of grid_id
fn index_to_pos((i, j): GridPos) -> Vec2 {
    Vec2::new(i as f32, j as f32)
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
