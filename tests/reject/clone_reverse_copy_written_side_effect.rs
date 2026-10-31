// expect: L0104
// The same through core's Reverse: Reverse<T>'s written clone calls T's,
// which prints here, so a container of it stays rejected too.
use std::cmp::Reverse;
#[derive(Copy, Debug)]
struct Loud(u8);
impl Clone for Loud {
    fn clone(&self) -> Loud {
        println!("clone called");
        *self
    }
}
fn main() {
    let v = vec![Reverse(Loud(1))];
    let w = v.clone();
    println!("{:?}", w.len());
}
