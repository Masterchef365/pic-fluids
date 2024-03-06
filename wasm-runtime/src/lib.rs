use bytemuck::{Pod, Zeroable};
use std::sync::Mutex;

static BUFFERS: Mutex<Buffers> = Mutex::new(Buffers::empty());

/// Dummy no-op kernel function
#[no_mangle]
fn per_particle_kernel(
    _dt: f32,
    _pos_x: f32,
    _pos_y: f32,
    _vel_x: f32,
    _vel_y: f32,
    _our_type: f32,
    out_accel_x: *mut f32,
    out_accel_y: *mut f32,
) {
    unsafe {
        *out_accel_x += 10.;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PerParticleInputPayload {
    dt: f32,
    pos: [f32; 2],
    vel: [f32; 2],
    our_type: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PerParticleOutputPayload {
    accel: [f32; 2],
}

unsafe impl Pod for PerParticleInputPayload {}
unsafe impl Zeroable for PerParticleInputPayload {}

unsafe impl Pod for PerParticleOutputPayload {}
unsafe impl Zeroable for PerParticleOutputPayload {}

/// Execute per_particle_kernel() for each of the populated input payloads
#[no_mangle]
fn run_per_particle_kernel() {
    let mut buf = BUFFERS.lock().unwrap();
    let (inp, outp) = buf.io();
    let inp: &[PerParticleInputPayload] = bytemuck::cast_slice(inp);
    let outp: &mut [PerParticleOutputPayload] = bytemuck::cast_slice_mut(outp);

    for (inp, outp) in inp.iter().zip(outp) {
        per_particle_kernel(
            inp.dt,
            inp.pos[0],
            inp.pos[1],
            inp.vel[0],
            inp.vel[1],
            inp.our_type,
            &mut outp.accel[0] as _,
            &mut outp.accel[1] as _,
        )
    }
}

pub struct Buffers {
    bytes: Vec<u8>,
    /// Beginning of output bytes, length of input bytes
    partition: usize,
}

/// Called first before either of the run_* functions;
/// reserves memory for two (contiguous) buffers and returns a pointer to the beginning of the
/// first
#[no_mangle]
pub fn reserve(input_bytes: usize, output_bytes: usize) -> usize {
    let mut buf = BUFFERS.lock().unwrap();
    *buf = Buffers::zeros(input_bytes, output_bytes);
    buf.ptr() as _
}

impl Buffers {
    pub const fn empty() -> Self {
        Self {
            bytes: vec![],
            partition: 0,
        }
    }

    pub fn zeros(input_bytes: usize, output_bytes: usize) -> Self {
        Self {
            partition: input_bytes,
            bytes: vec![0; input_bytes + output_bytes],
        }
    }

    pub fn ptr(&self) -> *const u8 {
        self.bytes.as_ptr()
    }

    pub fn io(&mut self) -> (&[u8], &mut [u8]) {
        let (inp, out) = self.bytes.split_at_mut(self.partition);
        (inp, out)
    }
}
