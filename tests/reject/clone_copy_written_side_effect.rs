// expect: L0104
// A Copy type with its own written Clone, cloned through a container: rustc
// calls the written clone (it prints), lust's glue would copy the bits, so
// lust rejects it rather than print something else (compute_glue_methods).
#[derive(Copy, Debug)]
struct Loud(u8);
impl Clone for Loud {
    fn clone(&self) -> Loud {
        println!("clone called");
        *self
    }
}
fn main() {
    let v = vec![Loud(1), Loud(2)];
    let w = v.clone();
    println!("{:?}", w);
}
