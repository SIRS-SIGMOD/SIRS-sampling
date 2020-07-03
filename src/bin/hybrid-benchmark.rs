
extern crate range_sampling;

use std::fs::File;
use std::io::BufReader;
use std::io::prelude::*;
use std::env;
use std::process;
use std::time::Instant;
use range_sampling::geo::Point;
use range_sampling::index::kdtree::KDTree;
use range_sampling::util::{Config, WPoint};

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: sampling_benchmark <config_file>");
        process::exit(-1);
    }
    let mut config_file = File::open(&args[1])?;
    let mut config_json = String::new();
    config_file.read_to_string(&mut config_json)?;
    let config: Config = serde_json::from_str(&config_json).expect("Config format error");

    let mut data: Vec<WPoint> = Vec::new();
    {
        let input_file = File::open(config.input_file)?;
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
                let w = nums[2].parse::<f64>().expect("Expect to be f64");
                data.push(WPoint{p: Point{x, y}, weight: w});
            }
        }
    }

    println!("Start building sampling indexes....");
    let now = Instant::now(); 
    let kdtree = KDTree::from(data);
    println!("Finish buildnig KD-Tree, takes {} s", now.elapsed().as_micros() as f64 / 1000000.0_f64);
    
    let period = 1;
    for i in 0..100 {
        std::thread::sleep(std::time::Duration::from_secs(1));
        kdtree.olken_range_sampling_throughput(&config.ranges[i], period);
        std::thread::sleep(std::time::Duration::from_secs(1));
        kdtree.range_sampling_throughput(&config.ranges[i], period);
        std::thread::sleep(std::time::Duration::from_secs(1));
        kdtree.range_sampling_no_rej_throughput(&config.ranges[i], period);
        std::thread::sleep(std::time::Duration::from_secs(1));
        kdtree.range_sampling_hybrid(&config.ranges[i], period);
    }
    Ok(())
}