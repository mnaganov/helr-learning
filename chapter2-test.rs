fn double<T: std::ops::Add<Output=T> + Copy>(x: T) -> T { x + x }
fn quadruple<T: std::ops::Add<Output=T> + Copy>(x: T) -> T { double(double(x)) }
fn factorial(n: u64) -> u64 { (1..=n).product() }
fn average(ns: &[i32]) -> i32 { (ns.iter().sum::<i32>() as i64 / ns.len() as i64) as i32 }
