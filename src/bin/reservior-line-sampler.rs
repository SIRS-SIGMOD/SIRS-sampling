use std::env;
use std::process;
use std::fs::File;
use std::io::BufReader;
use std::io::prelude::*;
use rand::distributions::{Uniform, Distribution};

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        eprintln!("Usage: reservior-line-sampler <input_file> <output_file> <sample_size>");
        process::exit(-1);
    }
    let sample_size: usize = (&args[3]).parse::<usize>().expect("<sample_size> should be a positive integer.");
    let input_file = File::open(&args[1])?;
    let mut buf_reader = BufReader::new(&input_file);
    let mut reservior: Vec<String> = Vec::new();
    let dis = Uniform::from(0.0f64..1.0f64);
    let mut rng = rand::thread_rng();
    let mut cnt: usize = 0;
    loop {
        let mut buf: String = String::new();
        let read_len = buf_reader.read_line(&mut buf)?;
        if read_len == 0 {
            break;
        } else {
            cnt += 1;
            if reservior.len() < sample_size {
                reservior.push(buf.clone());
            } else {
                let pos = (dis.sample(&mut rng) * cnt as f64) as usize;
                if pos < sample_size { reservior[pos] = buf.clone(); }
            }
        }
    }

    let mut output_file = File::create(&args[2])?;
    for line in reservior {
        output_file.write(&line.as_bytes())?;
    }
    Ok(())
}
