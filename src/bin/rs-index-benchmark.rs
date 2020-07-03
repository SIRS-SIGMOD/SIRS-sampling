
extern crate range_sampling;

use std::fs::File;
use std::io::BufReader;
use std::io::prelude::*;
use std::env;
use std::process;
use std::time::Instant;
use range_sampling::geo::Point;
use range_sampling::index::rstree::RSTree;
use range_sampling::util::WPoint;

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: rs-index-benchmark <input_file>");
        process::exit(-1);
    }

    let mut data: Vec<WPoint> = Vec::new();
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
                assert_eq!(nums.len(), 4);
                let x = nums[0].parse::<f64>().expect("Expect to be f64");
                let y = nums[1].parse::<f64>().expect("Expect to be f64");
                let z = nums[2].parse::<f64>().expect("Expect to be f64");
                let weight = nums[3].parse::<f64>().expect("Expect to be f64");
                data.push(WPoint{p: Point{x, y, z}, weight});
            }
        }
    }

    println!("Start building sampling indexes....");
    let now = Instant::now(); 
    let rstree = RSTree::from(&mut data);
    println!("Finish buildnig RS-Tree, takes {} s, index size = {}", now.elapsed().as_micros() as f64 / 1000000.0_f64, rstree.size());

    Ok(()) 
}