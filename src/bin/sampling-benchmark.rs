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
use range_sampling::index::rstree::RSTree;
use range_sampling::index::rsbtree::RSBTree;
use range_sampling::util::{Config, sample_from};

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

    let mut data: Vec<Point> = Vec::new();
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
                assert_eq!(nums.len(), 4);
                let x = nums[0].parse::<f64>().expect("Expect to be f64");
                let y = nums[1].parse::<f64>().expect("Expect to be f64");
                let z = nums[2].parse::<f64>().expect("Expect to be f64");
                data.push(Point{x, y, z});
            }
        }
    }

    let mut qts = false; let mut kds = false; let mut kdo = false;
    let mut kdb = false; let mut zvs = false; let mut rts = false;
    let mut rtb = false; let mut rto = false;
    for method in config.methods.iter() {
        if method == "qts" { qts = true; }
        else if method == "kds" { kds = true; }
        else if method == "kdo" { kdo = true; }
        else if method == "kdb" { kdb = true; }
        else if method == "zvs" { zvs = true; }
        else if method == "rts" { rts = true; }
        else if method == "rtb" { rtb = true; }
        else if method == "rto" { rto = true; }
    }
    println!("Start building sampling indexes....");
    let now = Instant::now(); 
    let kdtree = KDTree::from(&data);
    println!("Finish buildnig KD-Tree, takes {} s", now.elapsed().as_micros() as f64 / 1000000.0_f64);
    let rstree = if rts || rto {
        let now = Instant::now(); 
        let tree = RSTree::from(&data);
        println!("Finish buildnig RS-Tree, takes {} s", now.elapsed().as_micros() as f64 / 1000000.0_f64);
        Some(tree)
    } else { None };
    let mut kdbtree = if kdb {
        let now = Instant::now(); 
        let tree = KDBTree::from(&data);
        println!("Finish buildnig KD-Buffer-Tree, takes {} s", now.elapsed().as_micros() as f64 / 1000000.0_f64);
        Some(tree)
    } else { None };
    let mut rsbtree = if rtb {
        let now = Instant::now(); 
        let tree = RSBTree::from(&data);
        println!("Finish buildnig RS-Buffer-Tree, takes {} s", now.elapsed().as_micros() as f64 / 1000000.0_f64);
        Some(tree)
    } else { None };
    let zvtree = if zvs {
        let now = Instant::now(); 
        let tree = ZVTree::from(&data);
        println!("Finish buildnig ZV-Tree, takes {} s", now.elapsed().as_micros() as f64 / 1000000.0_f64);
        Some(tree)
    } else { None };

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
                let range_res = kdtree.range(&range);
                let samples = sample_from(&range_res, k.clone());
                tot_time += now.elapsed().as_micros();
                assert_eq!(samples.len(), k.clone());
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

    if let Some(rtree) = &rstree {
        if rts {
            for k in config.k_values.iter() {
                let mut tot_time: u128 = 0;
                for range in config.ranges.iter() {
                    let now = Instant::now();
                    let samples = rtree.range_sampling(&range, k.clone());
                    tot_time += now.elapsed().as_micros();
                    assert_eq!(samples.len(), k.clone());
                }
                let avg_latency = tot_time as f64 / config.ranges.len() as f64;
                println!("rts {} {} {}", avg_range_size, k, avg_latency);
            }
        }

        if rto {
            for k in config.k_values.iter() {
                let mut tot_time: u128 = 0;
                for range in config.ranges.iter() {
                    let now = Instant::now();
                    let samples = rtree.olken_range_sampling(&range, k.clone());
                    tot_time += now.elapsed().as_micros();
                    assert_eq!(samples.len(), k.clone());
                }
                let avg_latency = tot_time as f64 / config.ranges.len() as f64;
                println!("rto {} {} {}", avg_range_size, k, avg_latency);
            }
        }
    }
    
    if let Some(kbtree) = &mut kdbtree {
        for k in config.k_values.iter() {
            let mut tot_time: u128 = 0;
            for range in config.ranges.iter() {
                let now = Instant::now();
                let samples = kbtree.range_sampling(&range, k.clone());
                tot_time += now.elapsed().as_micros();
                assert_eq!(samples.len(), k.clone());
            }
            let avg_latency = tot_time as f64 / config.ranges.len() as f64;
            println!("kdb {} {} {}", avg_range_size, k, avg_latency);
        }
    }
    
    if let Some(rbtree) = &mut rsbtree {
        for k in config.k_values.iter() {
            let mut tot_time: u128 = 0;
            for range in config.ranges.iter() {
                let now = Instant::now();
                let samples = rbtree.range_sampling(&range, k.clone());
                tot_time += now.elapsed().as_micros();
                assert_eq!(samples.len(), k.clone());
            }
            let avg_latency = tot_time as f64 / config.ranges.len() as f64;
            println!("rtb {} {} {}", avg_range_size, k, avg_latency);
        }
    }
    
    if let Some(ztree) = &zvtree {
        for k in config.k_values.iter() {
            let mut tot_time: u128 = 0;
            for range in config.ranges.iter() {
                let now = Instant::now();
                let samples = ztree.range_sampling(&range, k.clone());
                tot_time += now.elapsed().as_micros();
                assert_eq!(samples.len(), k.clone());
            }
            let avg_latency = tot_time as f64 / config.ranges.len() as f64;
            println!("zvs {} {} {}", avg_range_size, k, avg_latency);
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
