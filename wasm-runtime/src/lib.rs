use bytemuck::{Pod, Zeroable};
use glam::Vec2;
use std::sync::Mutex;

// Required; tells the compiler to export definitions which the **node graph** will use
pub use builtins::*;
mod builtins;
pub mod query_accel;

static BUFFERS: Mutex<Buffers> = Mutex::new(Buffers::empty());

/// Dummy no-op kernel function
#[no_mangle] // No mangle so it can be found and replaced easily
#[inline(never)] // inline(never) required here for the function to be explicitly used by the run_* function!
extern "C" fn per_particle_kernel(
    out_ptr: *mut f32,
    dt: f32,
    ourtype: f32,
    position_x: f32,
    position_y: f32,
    velocity_x: f32,
    velocity_y: f32,
) {
    // Black boxes keep the compiler from deciding not to assign the function parameters
    std::hint::black_box(out_ptr);
    std::hint::black_box(dt);
    std::hint::black_box(ourtype);
    std::hint::black_box(position_x);
    std::hint::black_box(position_y);
    std::hint::black_box(velocity_x);
    std::hint::black_box(velocity_y);
}

/// Dummy no-op kernel function
#[no_mangle] // No mangle so it can be found and replaced easily
#[inline(never)] // inline(never) required here for the function to be explicitly used by the run_* function!
extern "C" fn per_neighbor_kernel(
    out_ptr: *mut f32,
    dt: f32,
    neighbor_radius: f32,
    ourtype: f32,
    theirtype: f32,
    pos_diff_x: f32,
    pos_diff_y: f32,
    position_x: f32,
    position_y: f32,
    velocity_x: f32,
    velocity_y: f32,
) {
    // Black boxes keep the compiler from deciding not to assign the function parameters
    std::hint::black_box(out_ptr);
    std::hint::black_box(dt);
    std::hint::black_box(ourtype);
    std::hint::black_box(theirtype);
    std::hint::black_box(neighbor_radius);
    std::hint::black_box(pos_diff_x);
    std::hint::black_box(pos_diff_y);
    std::hint::black_box(position_x);
    std::hint::black_box(position_y);
    std::hint::black_box(velocity_x);
    std::hint::black_box(velocity_y);
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PerParticleInputPayload {
    pub pos: [f32; 2],
    pub vel: [f32; 2],
    pub our_type: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PerParticleOutputPayload {
    pub accel: [f32; 2],
}

unsafe impl Pod for PerParticleInputPayload {}
unsafe impl Zeroable for PerParticleInputPayload {}

unsafe impl Pod for PerParticleOutputPayload {}
unsafe impl Zeroable for PerParticleOutputPayload {}

/// Execute per_particle_kernel() for each of the populated input payloads
#[no_mangle]
fn run_per_particle_kernel(dt: f32, neighbor_radius: f32) {
    let mut buf = BUFFERS.lock().unwrap();
    let (inp, outp) = buf.io();
    let inp: &[PerParticleInputPayload] = bytemuck::cast_slice(inp);
    let outp: &mut [PerParticleOutputPayload] = bytemuck::cast_slice_mut(outp);

    for (inp, outp) in inp.iter().zip(outp.iter_mut()) {
        per_particle_kernel(
            outp.accel.as_mut_ptr(),
            dt,
            inp.our_type,
            inp.pos[0],
            inp.pos[1],
            inp.vel[0],
            inp.vel[1],
        )
    }

    let points: Vec<Vec2> = inp.iter().map(|part| Vec2::from_array(part.pos)).collect();
    let accel = query_accel::QueryAccelerator::new(&points, neighbor_radius);

    let mut neigh_buf = vec![];

    for (i, out_part) in outp.iter_mut().enumerate() {
        neigh_buf.clear();
        neigh_buf.extend(accel.query_neighbors_fast(i, points[i]));
        for &neighbor in &neigh_buf {
            let diff = points[neighbor] - points[i];

            per_neighbor_kernel(
                out_part.accel.as_mut_ptr(),
                dt,
                neighbor_radius,
                inp[i].our_type,
                inp[neighbor].our_type,
                diff.x,
                diff.y,
                points[i].x,
                points[i].y,
                inp[i].vel[0],
                inp[i].vel[1],
            );
        }
    }
}

struct Buffers {
    bytes: Vec<u8>,
    /// Beginning of output bytes, length of input bytes
    partition: usize,
}

/// Called first before either of the run_* functions;
/// reserves memory for two (contiguous) buffers and returns a pointer to the beginning of the
/// first
#[no_mangle]
pub fn reserve(input_bytes: u32, output_bytes: u32) -> u32 {
    let mut buf = BUFFERS.lock().unwrap();
    *buf = Buffers::zeros(input_bytes as usize, output_bytes as usize);
    buf.ptr() as u32
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
