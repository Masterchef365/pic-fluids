use glam::Vec2;
use rand::prelude::*;

#[derive(Clone)]
pub struct Sim {
    pub protons: Vec<Vec2>,
    pub electrons: Vec<Vec2>,
}

impl Sim {
    pub fn new(
    ) -> Self {
        let mut rng = rand::thread_rng();

        Sim {
            protons: vec![Vec2::new(50., 50.)],
            electrons: vec![Vec2::new(30., 30.)],
        }
    }

    pub fn step(
        &mut self,
        dt: f32,
    ) {
    }
}
