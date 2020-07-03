use std::env;
use std::process;
use std::fs::File;
use std::io::BufReader;
use std::io::prelude::*;
use rand::distributions::{Uniform, Distribution};

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        eprintln!("Usage: rate-line-sampler <input_file> <output_file> <sample_rate>");
        process::exit(-1);
    }
    let sample_rate: f64 = (&args[3]).parse::<f64>().expect("<sample_rate> should be a f64.");
    if sample_rate > 1.0_f64 || sample_rate < 0.0_f64 {
        eprintln!("<sample_rate> invalid");
        process::exit(-1);
    }
    let input_file = File::open(&args[1])?;
    let mut output_file = File::create(&args[2])?;
    let mut buf_reader = BufReader::new(&input_file);
    let dis = Uniform::from(0.0f64..1.0f64);
    let mut rng = rand::thread_rng();
    let mut cnt: usize = 0;
    loop {
        let mut buf: String = String::new();
        let read_len = buf_reader.read_line(&mut buf)?;
        if read_len == 0 {
            break;
        } else {
            if dis.sample(&mut rng) <= sample_rate {
                cnt += 1;
                output_file.write(&buf.as_bytes())?;
            }
        }
    }

    println!("Sampling done: {} lines has been sampled.", cnt);
    Ok(())
}
