use std::rc::Rc;

use vorpal_wasm::CodeAnalysis;
use vorpal_widgets::vorpal_core::Node;
use wasm_bridge::{Engine, Instance, Module, Result, Store};
use wasm_runtime::{PerParticleInputPayload, PerParticleOutputPayload};

use crate::sim::{per_neighbor_fn_inputs, per_particle_fn_inputs};

pub struct WasmNodeRuntime {
    engine: Engine,
    store: Store<()>,
    instance: Instance,
    old_code: Option<(Rc<Node>, Rc<Node>)>,
}

const RUNTIME_WASM_BYTES: &[u8] =
    include_bytes!("../wasm-runtime/target/wasm32-unknown-unknown/release/wasm_runtime.wasm");
const PER_PARTICLE_RUN_FN_NAME: &str = "run_per_particle_kernel";

const PER_PARTICLE_KERNEL_FN_NAME: &str = "per_particle_kernel";
const PER_NEIGHBOR_KERNEL_FN_NAME: &str = "per_neighbor_kernel";

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
            old_code: None,
        })
    }

    pub fn update_code(&mut self, per_particle_node: &Rc<Node>, per_neighbor_node: &Rc<Node>) {
        // Don't bother resetting if there is no code change.
        // TODO: Find a way not to compare the entire graph...
        if self.old_code.as_ref() == Some(&(per_particle_node.clone(), per_neighbor_node.clone())) {
            return;
        }
        self.old_code = Some((per_particle_node.clone(), per_neighbor_node.clone()));

        // Innovative text-based linking technology
        let mut wat = wasmprinter::print_bytes(&RUNTIME_WASM_BYTES).unwrap();

        // Look for the first function declaration and insert the snippet just before that
        let idx = wat.find("(type").unwrap();

        // Compile to wasm binary
        for (node, name, input_types) in [
            (
                per_particle_node,
                PER_PARTICLE_KERNEL_FN_NAME,
                per_particle_fn_inputs(),
            ),
            (
                per_neighbor_node,
                PER_NEIGHBOR_KERNEL_FN_NAME,
                per_neighbor_fn_inputs(),
            ),
        ] {
            let anal = CodeAnalysis::new(node.clone(), &input_types);
            let nodes_wat_insert = anal.compile_function_to_wat(name).unwrap();
            eprintln!("{}", anal.func_name_rust(name).unwrap());

            // Rename the existing function to something else
            wat = wat.replacen(&format!("(func ${name}"), &format!("(func ${name}_old"), 1);
            wat.insert_str(idx, "\n");
            wat.insert_str(idx, &nodes_wat_insert);
        }

        let wasm = wat::parse_str(&wat).unwrap();
        self.set_code(&wasm).unwrap();
    }

    fn set_code(&mut self, code: &[u8]) -> Result<()> {
        let module = Module::new(&self.engine, code)?;
        self.instance = Instance::new(&mut self.store, &module, &[])?;
        Ok(())
    }

    pub fn run(
        &mut self,
        inputs: &[PerParticleInputPayload],
        dt: f32,
        neighbor_radius: f32,
    ) -> Result<Vec<PerParticleOutputPayload>> {
        // Casting
        let input_buf: &[u8] = bytemuck::cast_slice(inputs);
        let mut output_buf: Vec<PerParticleOutputPayload> =
            vec![PerParticleOutputPayload::default(); inputs.len()];

        // Reserve some memory in the wasm module
        let func = self
            .instance
            .get_typed_func::<(u32, u32), u32>(&mut self.store, "reserve")?;
        let output_buf_len = bytemuck::cast_slice::<_, u8>(&output_buf).len();

        let buf_ptr = func.call(
            &mut self.store,
            (input_buf.len() as u32, output_buf_len as u32),
        )?;
        let input_ptr = buf_ptr as usize;
        let output_ptr = input_ptr + input_buf.len();

        let mem = self.instance.get_memory(&mut self.store, "memory").unwrap();

        // Write input data
        mem.write(&mut self.store, input_ptr, &input_buf)?;

        // Call kernel run fn
        let func = self
            .instance
            .get_typed_func::<(f32, f32), ()>(&mut self.store, PER_PARTICLE_RUN_FN_NAME)?;
        func.call(&mut self.store, (dt, neighbor_radius))?;

        // Read results
        mem.read(
            &mut self.store,
            output_ptr,
            bytemuck::cast_slice_mut(&mut output_buf),
        )?;

        Ok(output_buf)
    }
}
