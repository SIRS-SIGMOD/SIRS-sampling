extern crate range_sampling;

use std::fs::File;
use std::io::BufReader;
use std::io::prelude::*;
use std::env;
use std::process;
use std::time::Instant;
use range_sampling::geo::{Point, MBR};
use range_sampling::index::kdtree::KDTree;
use range_sampling::index::rstree::RSTree;
use range_sampling::index::zvtree::ZVTree;
use range_sampling::util::WPoint;

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 5 {
        eprintln!("Usage: range-query-test <input_file> <query_file> <ratio> <area>");
        process::exit(-1);
    }

    let ratio = args[3].parse::<f64>().expect("Expect to be f64");
    assert!(ratio > 0.0);
    let area = args[4].parse::<f64>().expect("Expect to be f64");
    assert!(area > 0.0);
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
    let backup = data.clone();

    println!("Building Sampling Index....");
    let mut now = Instant::now(); 
    let zvtree = ZVTree::from(&data);
    println!("Finish buildnig ZVTree, takes {}", now.elapsed().as_micros() as f64 / 1000000.0_f64);
    now = Instant::now(); 
    let rstree = RSTree::from(&mut data);
    println!("Finish buildnig RSTree, takes {}", now.elapsed().as_micros() as f64 / 1000000.0_f64);
    now = Instant::now(); 
    let kdtree = KDTree::from(data);
    println!("Finish buildnig KDTree, takes {}", now.elapsed().as_micros() as f64 / 1000000.0_f64);

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
                assert_eq!(nums.len(), 4);
                let center_x = nums[0].parse::<f64>().expect("Expect to be f64");
                let center_y = nums[1].parse::<f64>().expect("Expect to be f64");
                let center_z = nums[2].parse::<f64>().expect("Expect to be f64");
                let width = (area / ratio).sqrt();
                let height = area / width;
                let time_span = 86400_f64;
                let query = MBR {
                    low: Point {
                        x: ((center_x - height) * 1e6).round() / 1e6,
                        y: ((center_y - width) * 1e6).round() / 1e6,
                        z: 0f64.max(center_z - time_span),
                    },
                    high: Point {
                        x: ((center_x + height) * 1e6).round() / 1e6,
                        y: ((center_y + width) * 1e6).round() / 1e6,
                        z: center_z + time_span,
                    },
                };
                let real_ans = backup.iter().filter(|item| query.contains(&item.p)).count();
                let count1 = zvtree.range(&query).len();
                let count2 = kdtree.range(&query).len();
                let count3 = rstree.range(&query).len();
                println!("Query MBR: {}", query);
                println!("{} {} {} {}", real_ans, count1, count2, count3);
                assert_eq!(count1, real_ans);
                assert_eq!(count2, real_ans);
                assert_eq!(count3, real_ans);
            }
        }
    }

    Ok(())
}
