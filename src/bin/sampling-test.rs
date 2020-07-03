extern crate range_sampling;

use std::fs::File;
use std::io::BufReader;
use std::io::prelude::*;
use std::env;
use std::process;
use std::time::Instant;
use std::collections::HashMap;
use range_sampling::geo::{Point, MBR};
use range_sampling::index::kdtree::KDTree;
use range_sampling::index::zvtree::ZVTree;
use range_sampling::util::WPoint;
//use range_sampling::index::rstree::RSTree;
//use range_sampling::index::kdbtree::KDBTree;

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: sampling-test <input_file>");
        process::exit(-1);
    }

    // [low: (-118.417606, 33.756715), high: (-118.297606, 33.876715)]
    // [low: (-117.520446, 47.58489), high: (-117.400446, 47.70489)]
    // [low: (-122.801398, 38.381212), high: (-122.681398, 38.501212)]

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
                let x = nums[1].parse::<f64>().expect("Expect to be f64");
                let y = nums[2].parse::<f64>().expect("Expect to be f64");
                let w = nums[3].parse::<f64>().expect("Expect to be f64");
                data.push(WPoint{p: Point{x, y}, weight: w});
            }
        }
    }

    let query = MBR {
        low: Point {
            x: -82.161485,
            y: 39.381252,
        },
        high: Point {
            x: -82.041485,
            y: 39.501252,
        },
    };

    // [low: (-122.801398, 38.381212), high: (-122.681398, 38.501212)]
    //let query = MBR {
    //    low: Point {
    //        x: -122.801398,
    //        y: 38.381212,
    //    },
    //    high: Point {
    //        x: -122.681398,
    //        y: 38.501212,
    //    },
    //};

    let zvtree = ZVTree::from(&data);
    let kdtree = KDTree::from(data);
    //let mut kdbtree = KDBTree::from(&mut data);
    //let rstree = RSTree::from(&mut data);
    {
        let now = Instant::now(); 
        let samples = kdtree.range_sampling(&query, 1000000);
        println!("KD-Sampling takes {} us", now.elapsed().as_micros());
        
        let mut map: HashMap<u64, usize> = HashMap::new();
        for sample in samples.iter() {
            let entry = map.entry(sample.to_zvalue()).or_insert(0);
            *entry += 1;
        }

        assert_eq!(map.len(), kdtree.range(&query).len());
        let dist: Vec<usize> = map.iter().map(|e| e.1.clone()).collect();
        println!("{:?}", dist);
    }

    println!("-----------------------------------------------------------"); 
    {
        let now = Instant::now(); 
        let samples = zvtree.range_sampling(&query, 1000000);
        println!("ZV-Sampling takes {} us", now.elapsed().as_micros());
        
        let mut map: HashMap<u64, usize> = HashMap::new();
        for sample in samples.iter() {
            let entry = map.entry(sample.to_zvalue()).or_insert(0);
            *entry += 1;
        }

        assert_eq!(map.len(), kdtree.range(&query).len());
        let dist: Vec<usize> = map.iter().map(|e| e.1.clone()).collect();
        println!("{:?}", dist);
    }

    /* 
    println!("-----------------------------------------------------------"); 
    {
        let now = Instant::now(); 
        let samples = kdbtree.range_sampling(&query, 1000000);
        println!("KDB-Sampling takes {} us", now.elapsed().as_micros());
        
        let mut map: HashMap<u64, usize> = HashMap::new();
        for sample in samples.iter() {
            let entry = map.entry(sample.to_zvalue()).or_insert(0);
            *entry += 1;
        }

        assert_eq!(map.len(), kdtree.range(&query).len());
        let dist: Vec<usize> = map.iter().map(|e| e.1.clone()).collect();
        println!("{:?}", dist);
    }
    */

    println!("-----------------------------------------------------------");
    {
        let now = Instant::now(); 
        let samples = kdtree.olken_range_sampling(&query, 1000000);
        println!("KD-Olken-Sampling takes {} us", now.elapsed().as_micros());
        
        let mut map: HashMap<u64, usize> = HashMap::new();
        for sample in samples.iter() {
            let entry = map.entry(sample.to_zvalue()).or_insert(0);
            *entry += 1;
        }

        assert_eq!(map.len(), kdtree.range(&query).len());
        let dist: Vec<usize> = map.iter().map(|e| e.1.clone()).collect();
        println!("{:?}", dist);
    }

    /* 
    println!("-----------------------------------------------------------"); 
    {
        let now = Instant::now(); 
        let samples = rstree.range_sampling(&query, 1000000);
        println!("RS-Sampling takes {} us", now.elapsed().as_micros());
        
        let mut map: HashMap<u64, usize> = HashMap::new();
        for sample in samples.iter() {
            let entry = map.entry(sample.to_zvalue()).or_insert(0);
            *entry += 1;
        }

        assert_eq!(map.len(), kdtree.range(&query).len());
        let dist: Vec<usize> = map.iter().map(|e| e.1.clone()).collect();
        println!("{:?}", dist);
    }
    */

    Ok(()) 
}
