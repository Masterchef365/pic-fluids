#[no_mangle]
pub extern "C" fn builtin_power(base: f32, exponent: f32) -> f32 {
    base.powf(exponent)
}

#[no_mangle]
pub extern "C" fn builtin_logbase(base: f32, value: f32) -> f32 {
    value.log(base)
}

#[no_mangle]
pub extern "C" fn builtin_cosine(value: f32) -> f32 {
    value.cos()
}

#[no_mangle]
pub extern "C" fn builtin_sine(value: f32) -> f32 {
    value.sin()
}

#[no_mangle]
pub extern "C" fn builtin_tangent(value: f32) -> f32 {
    value.tan()
}

#[no_mangle]
pub extern "C" fn builtin_natural_log(value: f32) -> f32 {
    value.ln()
}

#[no_mangle]
pub extern "C" fn builtin_natural_exp(value: f32) -> f32 {
    value.exp()
}

#[no_mangle]
pub extern "C" fn builtin_greater_than(lhs: f32, rhs: f32) -> f32 {
    f32::from(lhs > rhs)
}

#[no_mangle]
pub extern "C" fn builtin_less_than(lhs: f32, rhs: f32) -> f32 {
    f32::from(lhs < rhs)
}

