extern crate range_sampling;

use std::fs::File;
use std::io::BufReader;
use std::io::prelude::*;
use std::env;
use std::process;
use range_sampling::geo::Point;
use range_sampling::index::kdtree::KDTree;
use range_sampling::util::{SampleQuery, WPoint};

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 6 {
        eprintln!("Usage: range-count <input_file> <query_file> <ratio> <area> <k>");
        process::exit(-1);
    }

    let ratio = args[3].parse::<f64>().expect("Expect to be f64");
    assert!(ratio > 0.0);
    let area = args[4].parse::<f64>().expect("Expect to be f64");
    assert!(area > 0.0);
    let k = args[5].parse::<usize>().expect("Exepct to be usize");
    assert!(k > 0_usize);
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
                assert_eq!(nums.len(), 3);
                let x = nums[0].parse::<f64>().expect("Expect to be f64");
                let y = nums[1].parse::<f64>().expect("Expect to be f64");
                let w = nums[2].parse::<f64>().expect("Expect to be f64");
                data.push(WPoint{p: Point::new(x, y), weight: w});
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

    let kdtree = KDTree::from(data);
    for query in queries.iter() {
        let res = kdtree.range(&query.range);
        println!("{} {} {} {} {}", query.range.low.x, query.range.low.y, query.range.high.x, query.range.high.y, res.len());
    }
    

    Ok(()) 
}