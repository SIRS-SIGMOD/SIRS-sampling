
extern crate range_sampling;

use std::fs::File;
use std::io::BufReader;
use std::io::prelude::*;
use std::env;
use std::process;
use std::time::Instant;
use range_sampling::geo::Point;
use range_sampling::index::zvtree::ZVTree;
use range_sampling::util::Config;

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

    let mut data: Vec<(u64, f64)> = Vec::new();
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
                data.push((Point{x, y}.to_zvalue(), w));
            }
        }
    }

    println!("Start building sampling indexes....");
    let now = Instant::now(); 
    let zvtree = ZVTree::from_zvpoints(data);
    println!("Finish buildnig ZV-Tree, takes {} s", now.elapsed().as_micros() as f64 / 1000000.0_f64);

    let mut tot_range_size: usize = 0;
    for range in config.ranges.iter() {
        tot_range_size += zvtree.range(&range).len();
    }
    let avg_range_size = tot_range_size as f64 / config.ranges.len() as f64;

    for k in config.k_values.iter() {
        let mut tot_time: u128 = 0;
        for range in config.ranges.iter() {
            let now = Instant::now();
            let samples = zvtree.range_sampling(&range, k.clone());
            tot_time += now.elapsed().as_micros();
            assert_eq!(samples.len(), k.clone());
        }
        let avg_latency = tot_time as f64 / config.ranges.len() as f64;
        println!("zvs {} {} {}", avg_range_size, k, avg_latency);
    }

    Ok(()) 
}