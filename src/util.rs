use crate::geo;
use rand::distributions::{Uniform, Distribution};
use rand::SeedableRng;
use serde::{Serialize, Deserialize};

//type RNG = sfmt::SFMT;
type RNG = rand_pcg::Pcg64Mcg;
// type RNG = rand::rngs::StdRng;
// type RNG = rand::rngs::SmallRng;

#[inline(always)]
pub fn new_rng() -> impl rand::RngCore {
    RNG::from_entropy()
}

#[inline(always)]
pub fn approx_median<T>(data: &[T], k: usize, f : &dyn Fn(&T) -> f64) -> f64 {
    let mut rng = new_rng();
    let mut buffer: Vec<f64> = Vec::new();
    let dist = Uniform::from(0..data.len());
    for _ in 0..k {
        let offset = dist.sample(&mut rng);
        buffer.push(f(&data[offset]));
    }
    if k == 0 { f(&data[dist.sample(&mut rng)]) }
    else if k == 1 { buffer[0] }
    else {
        buffer.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        if k % 2 == 1 { buffer[k / 2] }
        else { (buffer[k / 2] + buffer[k/2 + 1]) / 2.0 }
    }
}

#[inline(always)]
pub fn partition_around<T>(points: &mut [T], f : &dyn Fn(&T) -> bool) -> usize {
    /*
    One pass partition algorithm.
    */
    let mut l_iter = 0;
    let mut r_iter = points.len() - 1;
    while l_iter < r_iter {
        if !f(& points[l_iter]) {
            points.swap(l_iter, r_iter);
            r_iter -= 1;
        } else {
            l_iter += 1;
        }
    }
    if f(& points[l_iter]) {
        l_iter + 1
    } else {
        l_iter
    }
}

#[inline(always)]
pub fn sample_from<T: Clone>(data: &[T], k: usize) -> Vec<T> {
    let mut samples: Vec<T> = Vec::new();
    let mut rng = new_rng();
    let dist = Uniform::from(0.0f64..1.0f64);
    let len = data.len();
    for _ in 0..k {
        samples.push(data[(dist.sample(&mut rng) * len as f64) as usize].clone());
    }
    samples
}

#[derive(Clone)]
pub struct WPoint {
    pub p: geo::Point,
    pub weight: f64,
}

pub fn wpoints_to_mbr(points: &[WPoint]) -> geo::MBR {
    let mut minx = std::f64::MAX;
    let mut miny = std::f64::MAX;
    let mut minz = std::f64::MAX;
    let mut maxx = std::f64::MIN;
    let mut maxy = std::f64::MIN;
    let mut maxz = std::f64::MIN;
    for p in points.iter() {
        minx = minx.min(p.p.x);
        miny = miny.min(p.p.y);
        minz = minz.min(p.p.z);
        maxx = maxx.max(p.p.x);
        maxy = maxy.max(p.p.y);
        maxz = maxz.max(p.p.z);
    }

    geo::MBR {
        low: geo::Point {
            x: minx,
            y: miny,
            z: minz,
        },
        high: geo::Point {
            x: maxx,
            y: maxy,
            z: maxz,
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct Config {
    pub input_file: String,
    pub ranges: Vec<geo::MBR>,
    pub k_values: Vec<usize>,
    pub methods: Vec<String>,
} 

pub struct SampleQuery {
    pub range: geo::MBR,
    pub k: usize,
}

const DAY_GAP_SEC: u32 = 86400 * 15;

impl SampleQuery {
    #[inline(always)]
    pub fn from(center_point: &geo::Point, area: f64, ratio: f64, k: usize) -> SampleQuery {
        let width = (area / ratio).sqrt();
        let height = area / width;
        let time_span = DAY_GAP_SEC as f64;
        let low_point = geo::Point {
            x: center_point.x - (height / 2.0_f64),
            y: center_point.y - (width  / 2.0_f64),
            z: 0.0f64.max(center_point.z - (time_span / 2.0_f64)),
        };
        let high_point = geo::Point {
            x: center_point.x + (height / 2.0_f64),
            y: center_point.y + (width  / 2.0_f64),
            z: center_point.z + (time_span / 2.0_f64),
        };
        SampleQuery {
            range: geo::MBR {
                low: low_point,
                high: high_point,
            },
            k,
        }
    }
}
