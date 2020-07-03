extern crate range_sampling;

use std::fs::File;
use std::io::BufReader;
use std::io::prelude::*;
use std::env;
use std::time::Instant;
use std::process;
use range_sampling::geo::Point;
use range_sampling::index::rstree::RSTree;
use range_sampling::util::SampleQuery;

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 6 {
        eprintln!("Usage: kd-sampling <input_file> <query_file> <ratio> <area> <k>");
        process::exit(-1);
    }

    let ratio = args[3].parse::<f64>().expect("Expect to be f64");
    assert!(ratio > 0.0);
    let area = args[4].parse::<f64>().expect("Expect to be f64");
    assert!(area > 0.0);
    let k = args[5].parse::<usize>().expect("Exepct to be usize");
    assert!(k > 0_usize);
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

    let mut queries: Vec<SampleQuery> = Vec::new();
    {
        let query_file = File::open(&args[2])?;
        let mut buf_reader = BufReader::new(&query_file);
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
                queries.push(SampleQuery::from(&Point{x, y}, area, ratio, k));
            }
        }
    }

    println!("Building Sampling Index....");
    let now = Instant::now(); 
    let rstree = RSTree::from(&mut data);
    println!("Finish buildnig index, takes {}", now.elapsed().as_micros() as f64 / 1000000.0_f64);

    let mut tot_time: u128 = 0;
    for query in queries.iter() {
        let now = Instant::now();
        let samples = rstree.range_sampling(&query.range, query.k);
        tot_time += now.elapsed().as_micros();
        assert_eq!(samples.len(), query.k);
    }

    let avg_latency = tot_time as f64 / queries.len() as f64;
    println!("Finish queries, total time: {}", tot_time as f64 / 1000000.0_f64);
    println!("Finish {} range sampling queries (k = {}), average query latency: {} us", queries.len(), k, avg_latency);

    Ok(()) 
}
