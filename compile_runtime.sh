mkdir bin
pushd wasm-runtime
cargo build --release
cp target/wasm32-unknown-unknown/release/wasm_runtime.wasm ../bin/
