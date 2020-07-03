extern crate range_sampling;

use std::fs::File;
use std::io::BufReader;
use std::io::prelude::*;
use std::env;
use std::process;
use std::time::Instant;
use range_sampling::geo::Point;
use range_sampling::util::Config;
use range_sampling::index::lsmtree::LSMTree;
use range_sampling::util::new_rng;
use rand::distributions::{Distribution, Uniform};

const WARMUP_COUNT: usize = 100000000;

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: sampling_benchmark <config_file> <insert_ratio>");
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
                assert_eq!(nums.len(), 3);
                let x = nums[0].parse::<f64>().expect("Expect to be f64");
                let y = nums[1].parse::<f64>().expect("Expect to be f64");
                data.push(Point{x, y});
            }
        }
    }

    let mut lsmtree = LSMTree::new(); 
    println!("Warming up by inserting {} points...", WARMUP_COUNT);
    let now = Instant::now();
    for i in 0..WARMUP_COUNT {
        lsmtree.insert(&data[i]);
    }
    println!("Finish warming up.... amortized insertion latency: {}us", now.elapsed().as_micros() as f64 / WARMUP_COUNT as f64);

    let insert_ratio = args[2].parse::<f64>().expect("ratio expect to be a f64");
    let mut rng = new_rng();
    let dist = Uniform::from(0.0f64..1.0f64);
    // let op_cnt = 10000;
    let mut op_cnt = 0;
    let mut query_cnt = 0;
    let mut insert_cnt = 0;
    let mut insert_time: u128 = 0;
    let mut query_time: u128 = 0;
    while op_cnt < 100000 {
        let coin = dist.sample(&mut rng);
        if coin < insert_ratio {
            let timer = Instant::now();
            lsmtree.insert(&data[insert_cnt + WARMUP_COUNT]);
            insert_time += timer.elapsed().as_micros();
            insert_cnt += 1;
        } else {
            let timer = Instant::now();
            lsmtree.range_sampling(&config.ranges[query_cnt], 1000);
            query_time += timer.elapsed().as_micros();
            query_cnt += 1;
        }
        op_cnt += 1;
        if query_cnt == config.ranges.len() { break; }
    }
    if insert_cnt > 0 { println!("Done {} insertions, with amortized insertion latency: {}us", insert_cnt, insert_time as f64 / insert_cnt as f64); }
    if query_cnt > 0 { println!("Done {} queries, with amortized query latency: {}us", query_cnt, query_time as f64 / query_cnt as f64); }

    Ok(()) 
}
