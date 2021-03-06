extern crate range_sampling;

use std::fs::File;
use std::io::BufReader;
use std::io::prelude::*;
use std::env;
use std::process;
use std::time::Instant;
use range_sampling::geo::Point;
use range_sampling::index::kdtree::KDTree;
use range_sampling::index::kdbtree::KDBTree;
use range_sampling::index::zvtree::ZVTree;

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: index_benchmark <input_file>");
        process::exit(-1);
    }

    let mut data: Vec<Point> = Vec::new();
    {
        let input_file = File::open(&args[1])?;
        let mut buf_reader = BufReader::new(&input_file);
        let mut buf: String = String::new();
        loop {
            buf.clear();
            let read_len = buf_reader.read_line(&mut buf)?;
            if read_len == 0 {
                break;
            } else {
                let nums: Vec<&str> = buf[0..buf.len() - 1].split(' ').collect();
                assert_eq!(nums.len(), 3);
                let x = nums[0].parse::<f64>().expect("Expect to be f64");
                let y = nums[1].parse::<f64>().expect("Expect to be f64");
                data.push(Point{x, y});
            }
        }
    }

    println!("Start building sampling indexes....");
    let now = Instant::now(); 
    let kdtree = KDTree::from(&data);
    println!("Finish buildnig KD-Tree, takes {} s, index size = {}", now.elapsed().as_micros() as f64 / 1000000.0_f64, kdtree.size());
    let now = Instant::now(); 
    let kdbtree = KDBTree::from(&data);
    println!("Finish buildnig KD-Buffer-Tree, takes {} s, index size = {}", now.elapsed().as_micros() as f64 / 1000000.0_f64, kdbtree.size());
    let now = Instant::now(); 
    let zvtree = ZVTree::from(&data);
    println!("Finish buildnig ZV-Tree, takes {} s, index size = {}", now.elapsed().as_micros() as f64 / 1000000.0_f64, zvtree.size());

    Ok(()) 
}