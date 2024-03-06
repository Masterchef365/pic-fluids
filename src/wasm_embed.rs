use wasm_bridge::{Result, Engine, Store, Module, Instance};
use wasm_runtime::{PerParticleOutputPayload, PerParticleInputPayload};

pub struct WasmNodeRuntime {
    engine: Engine,
    store: Store<()>,
    instance: Instance,
}

const RUNTIME_WASM_BYTES: &[u8] = include_bytes!("../wasm-runtime/target/wasm32-unknown-unknown/release/wasm_runtime.wasm");

impl WasmNodeRuntime {
    pub fn new() -> Result<Self> {
        let engine = Engine::new(&Default::default())?;
        let mut store = Store::new(&engine, ());

        let module = Module::new(&engine, RUNTIME_WASM_BYTES)?;
        let instance = Instance::new(&mut store, &module, &[])?;

        Ok(Self {
            instance,
            engine,
            store,
        })
    }

    pub fn run(&mut self, inputs: &[PerParticleInputPayload]) -> Result<Vec<PerParticleOutputPayload>> {
        // Casting
        let input_buf: &[u8] = bytemuck::cast_slice(inputs);
        let mut output_buf: Vec<PerParticleOutputPayload> = vec![PerParticleOutputPayload::default(); inputs.len()];

        // Reserve some memory in the wasm module
        let func = self.instance.get_typed_func::<(u32, u32), u32>(&mut self.store, "reserve")?;
        let buf_ptr = func.call(&mut self.store, (input_buf.len() as u32, output_buf.len() as u32))?;
        let input_ptr = buf_ptr as usize;
        let output_ptr = input_ptr + input_buf.len();

        let mem = self.instance.get_memory(&mut self.store, "memory").unwrap();

        // Write input data
        mem.write(&mut self.store, input_ptr, &input_buf)?;

        // Call kernel run fn
        let func = self.instance.get_typed_func::<(), ()>(&mut self.store, "run_per_particle_kernel")?;
        func.call(&mut self.store, ())?;

        // Read results
        mem.read(&mut self.store, output_ptr, bytemuck::cast_slice_mut(&mut output_buf))?;

        Ok(output_buf)
    }
}
