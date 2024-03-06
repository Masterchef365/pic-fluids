#[no_mangle]
fn add(a: i32, b: i32) -> i32 {
    a + b
}

/*
use std::sync::Mutex;

pub struct Buffers {
    bytes: Vec<u8>,
    /// Beginning of output bytes, length of input bytes
    partition: usize,
}

static BUFFERS: Mutex<Buffers> = Mutex::new(Buffers::empty());

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

    pub fn input_bytes(&self) -> &[u8] {
        &self.bytes[..self.partition]
    }

    pub fn output_bytes(&mut self) -> &mut [u8] {
        &mut self.bytes[self.partition..]
    }
}
*/
