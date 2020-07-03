use std::fmt;
use serde::{Serialize, Deserialize};

const RESOLUTION_X: f64 = 1e6;
const RESOLUTION_Y: f64 = 1e6;
const BASE_X: i32 = 180_000_000;
const BASE_Y: i32 = 90_000_000;

#[derive(PartialEq, Clone, Serialize, Deserialize, Debug)]
pub struct Point {
    pub x: f64, //lng
    pub y: f64, //lat
}

#[derive(PartialEq, Clone, Serialize, Deserialize, Debug)]
pub struct MBR {
    pub low: Point,
    pub high: Point,
}

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

impl fmt::Display for MBR {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[low: {}, high: {}]", self.low, self.high)
    }
}

impl Point {
    pub fn new(x_: f64, y_: f64) -> Point {
        Point { x: x_, y: y_ }
    }

    pub fn from_zvalue(zv: u64) -> Point {
        let mut tmpx = 0 as u32;
        let mut tmpy = 0 as u32;
        let mut tmp = zv;
        for i in 0..32 {
            tmpx += ((tmp & 2) >> 1 << i) as u32;
            tmpy += ((tmp & 1) << i) as u32;
            tmp = tmp >> 2;
        }
        Point {
            x: (tmpx as i32 - BASE_X) as f64 / RESOLUTION_X,
            y: (tmpy as i32 - BASE_Y) as f64 / RESOLUTION_Y,
        }
    }

    pub fn zvalue_to_raw(zv: u64) -> (u32, u32) {
        let mut tmpx = 0 as u32;
        let mut tmpy = 0 as u32;
        let mut tmp = zv;
        for i in 0..32 {
            tmpx += ((tmp & 2) >> 1 << i) as u32;
            tmpy += ((tmp & 1) << i) as u32;
            tmp = tmp >> 2;
        }
        (tmpx, tmpy)
    }

    pub fn to_zvalue(&self) -> u64 {
        let mut tmpx: u32 = ((self.x * RESOLUTION_X) as i32 + BASE_X) as u32;
        let mut tmpy: u32 = ((self.y * RESOLUTION_Y) as i32 + BASE_Y) as u32;
        let mut res: u64 = 0;
        for i in 0..32 {
            res += ((((tmpx & 1) << 1) + (tmpy & 1)) as u64) << (i * 2);
            tmpx = tmpx >> 1;
            tmpy = tmpy >> 1;
        }
        res
    }
    
    pub fn get_scaled(&self) -> (u32, u32) {
        (((self.x * RESOLUTION_X) as i32 + BASE_X) as u32,  ((self.y * RESOLUTION_Y) as i32 + BASE_Y) as u32)
    }

    pub fn compose_zvalue(x: u32, y: u32) -> u64 {
        let mut tmpx = x;
        let mut tmpy = y;
        let mut res: u64 = 0;
        for i in 0..32 {
            res += ((((tmpx & 1) << 1) + (tmpy & 1)) as u64) << (i * 2);
            tmpx = tmpx >> 1;
            tmpy = tmpy >> 1;
        }
        res
    }

    pub fn min_dist(&self, other: &Point) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }

    pub fn min_dist_mbr(&self, other: &MBR) -> f64 {
        let mut ans = 0.0_f64;
        if self.x < other.low.x { ans += (other.low.x - self.x) * (other.low.x - self.x) }
        else if self.x > other.high.x { ans += (self.x - other.high.x) * (self.x - other.high.x)}
        if self.y < other.low.y { ans += (other.low.y - self.y) * (other.low.y - self.y) }
        else if self.y > other.high.y { ans += (self.y - other.high.y) * (self.y - other.high.y)}
        ans
    }
}

impl MBR {
    pub fn new(low_: &Point, high_: &Point) -> MBR {
        MBR {
            low: Point {
                x: low_.x,
                y: low_.y,
            },
            high: Point {
                x: high_.x,
                y: high_.y,
            },
        }
    }

    pub fn from_points(points: &[Point]) -> MBR {
        let mut minx = std::f64::MAX;
        let mut miny = std::f64::MAX;
        let mut maxx = std::f64::MIN;
        let mut maxy = std::f64::MIN;
        for p in points.iter() {
            minx = minx.min(p.x);
            miny = miny.min(p.y);
            maxx = maxx.max(p.x);
            maxy = maxy.max(p.y);
        }

        MBR {
            low: Point {
                x: minx,
                y: miny,
            },
            high: Point {
                x: maxx,
                y: maxy,
            }
        }
    }

    pub fn contains(&self, p: &Point) -> bool {
        p.x >= self.low.x && p.x <= self.high.x && p.y >= self.low.y && p.y <= self.high.y
    }

    pub fn contains_mbr(&self, other: &MBR) -> bool {
        self.low.x <= other.low.x && self.low.y <= other.low.y &&
        self.high.x >= other.high.x && self.high.y >= other.high.y
    }

    pub fn intersects(&self, other: &MBR) -> bool {
        !(self.low.x > other.high.x || self.high.x < other.low.x ||
          self.low.y > other.high.y || self.high.y < other.low.y)
    }
}
