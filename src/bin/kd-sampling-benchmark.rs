extern crate range_sampling;

use std::fs::File;
use std::io::BufReader;
use std::io::prelude::*;
use std::env;
use std::process;
use std::time::Instant;
use range_sampling::geo::Point;
use range_sampling::index::kdtree::KDTree;
use range_sampling::util::{Config, WPoint, new_rng};
use range_sampling::alias::AliasTable;
use rand::distributions::{Uniform, Distribution};

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

    let mut qts = false; let mut kds = false; let mut kdo = false;
    for method in config.methods.iter() {
        if method == "qts" { qts = true; }
        else if method == "kds" { kds = true; }
        else if method == "kdo" { kdo = true; }
    }
    println!("Start building sampling indexes....");
    let now = Instant::now(); 
    let kdtree = KDTree::from(data);
    println!("Finish buildnig KD-Tree, takes {} s", now.elapsed().as_micros() as f64 / 1000000.0_f64);

    let mut tot_range_size: usize = 0;
    for range in config.ranges.iter() {
        tot_range_size += kdtree.range(&range).len();
    }
    let avg_range_size = tot_range_size as f64 / config.ranges.len() as f64;

    if qts {
        for k in config.k_values.iter() {
            let mut tot_time: u128 = 0;
            for range in config.ranges.iter() {
                let now = Instant::now();
                let res = kdtree.range(&range);
                let weights: Vec<f64> = res.iter().map(|p| p.weight).collect();
                let alias = AliasTable::from(&weights);
                let mut samples: Vec<Point> = Vec::new();
                let mut rng = new_rng();
                let dist = Uniform::from(0.0f64..1.0f64);
                for _ in 0..*k {
                    let coin1 = dist.sample(&mut rng);
                    let coin2 = dist.sample(&mut rng);
                    samples.push(res[alias.sample(coin1, coin2)].p.clone());
                }
                tot_time += now.elapsed().as_micros();
                assert_eq!(samples.len(), *k);
            }
            let avg_latency = tot_time as f64 / config.ranges.len() as f64;
            println!("qts {} {} {}", avg_range_size, k, avg_latency);
        }
    }
    
    if kds {
        for k in config.k_values.iter() {
            let mut tot_time: u128 = 0;
            for range in config.ranges.iter() {
                let now = Instant::now();
                let samples = kdtree.range_sampling(&range, k.clone());
                tot_time += now.elapsed().as_micros();
                assert_eq!(samples.len(), k.clone());
            }
            let avg_latency = tot_time as f64 / config.ranges.len() as f64;
            println!("kds {} {} {}", avg_range_size, k, avg_latency);
        }
    }
    
    if kdo {
        for k in config.k_values.iter() {
            let mut tot_time: u128 = 0;
            for range in config.ranges.iter() {
                let now = Instant::now();
                let samples = kdtree.olken_range_sampling(&range, k.clone());
                tot_time += now.elapsed().as_micros();
                assert_eq!(samples.len(), k.clone());
            }
            let avg_latency = tot_time as f64 / config.ranges.len() as f64;
            println!("kdo {} {} {}", avg_range_size, k, avg_latency);
        }
    }
    Ok(()) 
}
