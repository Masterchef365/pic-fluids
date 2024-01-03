use glam::Vec2;
use rand::prelude::*;
use rand_distr::Normal;

/// Parameters which may change between frames
#[derive(Clone, Debug)]
pub struct SimTweaks {
    pub proton_dt: f32,
    pub proton_proton_smooth: f32,
    pub proton_electron_smooth: f32,
    pub bohr_radius: f32,

    pub electron_steps: usize,
    pub electron_temperature: f32,
    pub electron_sigma: f32,
    pub electron_electron_smooth: f32,
    pub electron_proton_smooth: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct Electron {
    pub pos: Vec2,
}

#[derive(Clone, Copy, Debug)]
pub struct Proton {
    pub pos: Vec2,
    pub vel: Vec2,
}

#[derive(Clone)]
pub struct Sim {
    pub protons: Vec<Proton>,
    pub electrons: Vec<Electron>,
}

impl Sim {
    pub fn new(n_protons: usize, n_electrons: usize) -> Self {
        let mut rng = rand::thread_rng();

        let mut electrons = vec![];
        for _ in 0..n_electrons {
            electrons.push(Electron {
                pos: Vec2::new(rng.gen_range(0.0..100.0), rng.gen_range(0.0..100.0)),
            });
        }

        let mut protons = vec![];
        for _ in 0..n_protons {
            protons.push(Proton {
                pos: Vec2::new(rng.gen_range(25.0..75.0), rng.gen_range(25.0..75.0)),
                vel: Vec2::ZERO,
            });
        }

        Sim { protons, electrons }
    }

    /// Returns the history of the electron cloud during this step
    pub fn step(&mut self, tweak: &SimTweaks) -> Vec<Electron> {
        let mut rng = rand::thread_rng();

        // Propose a new configuration for electrons
        let mut cloud = self.electrons.clone();
        for _ in 0..tweak.electron_steps * self.electrons.len() {
            let idx = rng.gen_range(0..self.electrons.len());
            let old_u = self.electron_potential_energy(idx, tweak);
            let old_pos = self.electrons[idx].pos;

            let normal = Normal::new(0., tweak.electron_sigma).unwrap();
            let new_pos = old_pos + Vec2::new(normal.sample(&mut rng), normal.sample(&mut rng));

            // TODO: Proper clamping
            let new_pos = new_pos.clamp(Vec2::ZERO, Vec2::splat(100.));

            self.electrons[idx].pos = new_pos;
            let new_u = self.electron_potential_energy(idx, tweak);

            let du = new_u - old_u;
            let rate = (-du / tweak.electron_temperature).exp().min(0.99);

            if !rng.gen_bool(rate as _) {
                // Reject
                self.electrons[idx].pos = old_pos;
            } else {
                cloud.push(self.electrons[idx]);
            }
        }

        // Update proton positions
        let old_pos = self.protons.clone();

        for idx in 0..self.protons.len() {
            let pos = old_pos[idx].pos;

            let mut force = Vec2::ZERO;

            // Accumulate proton forces
            for other_idx in 0..self.protons.len() {
                if idx != other_idx {
                    let diff = old_pos[other_idx].pos - pos;
                    force -= smoothed_force(diff, tweak.proton_proton_smooth);
                }
            }

            // Accumulate electron forces
            for elect in &cloud {
                let diff = elect.pos - pos;
                force += smoothed_force(diff, tweak.proton_electron_smooth)
                    / tweak.electron_steps as f32;
            }

            // Step proton
            self.protons[idx].vel += force;
            let vel = self.protons[idx].vel;
            self.protons[idx].pos += vel * tweak.proton_dt;
        }

        cloud
    }

    fn electron_potential_energy(&self, idx: usize, tweak: &SimTweaks) -> f32 {
        let mut u = 0.0;

        let pos = self.electrons[idx].pos;

        for i in 0..self.electrons.len() {
            if i != idx {
                u += smoothed_potential(
                    self.electrons[i].pos.distance(pos),
                    tweak.electron_electron_smooth,
                    0.,
                )
            }
        }

        for prot in &self.protons {
            u -= smoothed_potential(
                prot.pos.distance(pos),
                tweak.electron_proton_smooth,
                tweak.bohr_radius,
            );
        }

        u
    }
}

fn smoothed_potential(r: f32, smooth: f32, bohr_radius: f32) -> f32 {
    1. / ((r - bohr_radius).abs() + smooth)
}

fn smoothed_force(r: Vec2, smooth: f32) -> Vec2 {
    r.normalize() / (r.length_squared() + smooth)
}

impl Default for SimTweaks {
    fn default() -> Self {
        Self {
            // The proton is about 1800 times heavier than the electron...
            proton_dt: 1e-3,
            proton_proton_smooth: 0.0,
            proton_electron_smooth: 1.0,
            bohr_radius: 2.0,

            electron_steps: 1000,
            electron_sigma: 1.0,
            electron_temperature: 5e-2,
            electron_proton_smooth: 1.0,
            electron_electron_smooth: 0.0,
        }
    }
}
