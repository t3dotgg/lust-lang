// Clone of containers whose element has a written Clone next to a Copy impl
// whose bounds do not hold, or core's Copy `Reverse<T>`:
// `Vec`, `Option`, and `VecDeque` of them clone (they were "no method `clone`").
use std::cmp::Reverse;
use std::collections::VecDeque;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct P(u8, u16);

// Copy only when T is; a written Clone with a side effect. For a non-Copy T
// rustc calls it through the container, so the glue must call it too.
#[derive(Copy, Debug)]
struct R<T>(T);
impl<T: Clone> Clone for R<T> {
    fn clone(&self) -> R<T> {
        println!("R::clone");
        R(self.0.clone())
    }
}

fn main() {
    let v = vec![Reverse(3u32), Reverse(1), Reverse(2)];
    let mut w = v.clone();
    w.sort();
    println!("{:?} {:?}", v, w);

    let o: Option<Reverse<P>> = Some(Reverse(P(1, 2)));
    println!("{:?}", o.clone());

    let mut d: VecDeque<Reverse<(u32, i64)>> = VecDeque::new();
    d.push_back(Reverse((1, -2)));
    d.push_front(Reverse((4, 5)));
    println!("{:?} {}", d.clone(), d.len());

    let s: Vec<Reverse<String>> = vec![Reverse(String::from("a")), Reverse(String::from("b"))];
    println!("{:?}", s.clone());

    let r = vec![R(String::from("x")), R(String::from("y"))];
    println!("{:?}", r.clone());
    let ro = Some(R(String::from("z")));
    println!("{:?}", ro.clone());
}
