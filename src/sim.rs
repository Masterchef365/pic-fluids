use glam::Vec2;
use rand::prelude::*;
use rand_distr::Normal;

/// Parameters which may change between frames
#[derive(Clone, Debug)]
pub struct SimTweaks {
    pub proton_dt: f32,
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
    pub fn new() -> Self {
        let mut rng = rand::thread_rng();

        Sim {
            protons: vec![Proton {
                pos: Vec2::new(50., 50.),
                vel: Vec2::ZERO,
            }],
            electrons: vec![Electron {
                pos: Vec2::new(30., 30.),
            }],
        }
    }

    pub fn step(&mut self, tweak: &SimTweaks) {
        let mut rng = rand::thread_rng();

        // Propose a new configuration for electrons
        for _ in 0..tweak.electron_steps {
            let idx = rng.gen_range(0..self.electrons.len());
            let old_u = self.electron_potential_energy(idx, tweak);
            let old_pos = self.electrons[idx].pos;

            let normal = Normal::new(0., tweak.electron_sigma).unwrap();
            let new_pos = old_pos + Vec2::new(normal.sample(&mut rng), normal.sample(&mut rng));
            self.electrons[idx].pos = new_pos;
            let new_u = self.electron_potential_energy(idx, tweak);

            let du = new_u - old_u;
            let rate = (-du / tweak.electron_temperature).exp().min(0.99);

            dbg!(rate);

            if !rng.gen_bool(rate as _) {
                self.electrons[idx].pos = old_pos;
            }
        }
    }

    fn electron_potential_energy(&self, idx: usize, tweak: &SimTweaks) -> f32 {
        let mut u = 0.0;

        let pos = self.electrons[idx].pos;

        for i in 0..self.electrons.len() {
            if i != idx {
                u += smoothed_potential(self.electrons[i].pos.distance(pos), tweak.electron_electron_smooth)
            }
        }

        for prot in &self.protons {
            u -= smoothed_potential(prot.pos.distance(pos), tweak.electron_proton_smooth);
        }

        u
    }
}

fn smoothed_potential(r: f32, smooth: f32) -> f32 {
    1. / (r.abs() + smooth)
}

impl Default for SimTweaks {
    fn default() -> Self {
        Self {
            proton_dt: 1e-2,
            electron_steps: 10,
            electron_sigma: 0.05,
            electron_temperature: 1e-4,
            electron_proton_smooth: 1.0,
            electron_electron_smooth: 1.0,
        }
    }
}
