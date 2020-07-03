use crate::index::zvtree::ZVTree;
use crate::geo::{Point, MBR};
use crate::alias::AliasTable;
use crate::util;
use rand::distributions::{Distribution, Uniform};

const TOP_LEVEL_CAPACITY: usize = 640;

pub struct LSMTree {
    top_level: Vec<u64>,
    levels: Vec<Option<ZVTree>>,
}

impl LSMTree {
    pub fn new() -> LSMTree {
        LSMTree {
            top_level: Vec::new(),
            levels: Vec::new(),
        }
    }

    pub fn insert(&mut self, p: &Point) {
        self.top_level.push(p.to_zvalue());
        if self.top_level.len() == TOP_LEVEL_CAPACITY {
            self.merge_top_level()
        }
    }

    pub fn print_levels(&self) {
        println!("{}", self.top_level.len());
        for i in 0..self.levels.len() {
            println!("{}", self.get_level_len(i))
        }
    }
    
    pub fn check_bound(zv: u64, lowx: u32, lowy: u32, highx: u32, highy: u32) -> bool {
        let (x, y) = Point::zvalue_to_raw(zv);
        x >= lowx && x <= highx && y >= lowy && y <= highy
    }

    pub fn range_sampling(&self, query: &MBR, k: usize) -> Vec<Point> {
        let (lowx, lowy) = query.low.get_scaled();
        let (highx, highy) = query.high.get_scaled();
        let mut samples: Vec<Point> = Vec::new();
        let mut top_level_res: Vec<Point> = Vec::new();
        for p in self.top_level.iter() {
            if LSMTree::check_bound(*p, lowx, lowy, highx, highy) {
                top_level_res.push(Point::from_zvalue(*p));
            }
        }
        let mut aliases: Vec<(usize, Option<AliasTable>, Vec<(usize, usize)>)> = Vec::new();
        let mut weights: Vec<f64> = Vec::new();
        let mut tot_size = 0;
        for level in self.levels.iter() {
            match level {
                Some(tree) => {
                    let tmp = tree.get_top_level_alias(query);
                    weights.push(tmp.0 as f64);
                    tot_size += tmp.0;
                    aliases.push(tmp);
                }
                None => {
                    aliases.push((0, None, Vec::new()));
                    weights.push(0f64);
                }
            }
        }
        weights.push(top_level_res.len() as f64);
        let top_level_alias = AliasTable::from(&weights);
        if top_level_res.len() + tot_size > 0 {
            let mut rng = util::new_rng();
            let dist = Uniform::from(0.0f64..1.0f64);
            while samples.len() < k {
                let coin1 = dist.sample(&mut rng);
                let coin2 = dist.sample(&mut rng);
                let res = top_level_alias.sample(coin1, coin2);
                if res == aliases.len() {
                    samples.push(top_level_res[(dist.sample(&mut rng) * (top_level_res.len() as f64)) as usize].clone());
                } else {
                    if let Some(alias) = &aliases[res].1 {
                        let coin3 = dist.sample(&mut rng);
                        let coin4 = dist.sample(&mut rng);
                        let tmp = alias.sample(coin3, coin4);
                        let range = aliases[res].2[tmp];
                        let offset = ((range.1 - range.0) as f64 * dist.sample(&mut rng)) as usize + range.0;
                        if let Some(tree) = &self.levels[res] {
                            if tree.check_bound(offset, lowx, lowy, highx, highy) {
                                samples.push(Point::from_zvalue(tree.data[offset]))
                            }
                        }
                    }
                }
            }
        } 
        samples
    }


    fn merge(left: &[u64], right: &[u64]) -> Vec<u64> {
        let merged_len = left.len() + right.len();
        let mut res: Vec<u64> = Vec::with_capacity(merged_len);
        let mut i = 0;
        let mut j = 0;
        while res.len() < merged_len {
            if i == left.len() {
                res.push(right[j]);
                j += 1;
            } else if j == right.len() {
                res.push(left[i]);
                i += 1;
            } else if right[j] < left[i] {
                res.push(right[j]);
                j += 1;
            } else {
                res.push(left[i]);
                i += 1;
            }
        }
       res 
    }

    fn merge_top_level(&mut self) {
        let mut current_level = 0;
        self.top_level.sort_unstable();
        let mut merged: Vec<u64> = self.top_level.clone();
        if !self.levels.is_empty() {
            if let Some(level) = &self.levels[0] {
                merged = LSMTree::merge(&merged, &level.data)
            }
        }
        while current_level < self.levels.len() && self.top_level.len() + self.get_level_len(current_level) > LSMTree::level_capacity(current_level) {
            self.levels[current_level] = None;
            current_level += 1;
            if current_level == self.levels.len() { break; }
            if let Some(level) = &self.levels[current_level] {
                merged = LSMTree::merge(&merged, &level.data)
            }
        }
        if current_level == self.levels.len() {
            self.levels.push(Some(ZVTree::from_sorted_zvs(merged)));
        } else {
            self.levels[current_level] = Some(ZVTree::from_sorted_zvs(merged));
        }
        self.top_level.clear();
    }

    fn get_level_len(&self, level: usize) -> usize {
        match &self.levels[level] {
            Some(l) => { l.len() }
            None => { 0 }
        }
    }

    fn level_capacity(level: usize) -> usize {
        (1 << (level + 1)) * TOP_LEVEL_CAPACITY
    }
}