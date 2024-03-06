use wasm_bridge::{Result, Engine, Store, Module, Instance};

pub struct WasmNodeRuntime {
    engine: Engine,
    store: Store<()>,
    instance: Instance,
}

const RUNTIME_WASM_BYTES: &[u8] = include_bytes!("../target/wasm32-unknown-unknown/release/wasm_runtime.wasm");

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

    pub fn run(&mut self) -> Result<i32> {
        let func = self.instance.get_typed_func::<(i32, i32), i32>(&mut self.store, "add")?;
        let res = func.call(&mut self.store, (4, 5))?;

        Ok(res)
    }
}
