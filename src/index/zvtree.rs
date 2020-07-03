use crate::alias::AliasTable;
use crate::geo::{MBR, Point};
use crate::util;
use superslice::*;
use std::rc::Rc;
use rand::distributions::{Uniform, Distribution};

const MAX_ENTRIES_PER_NODE: usize = 256;

struct ZVTreeNode {
    children: Option<(Rc<ZVTreeNode>, Rc<ZVTreeNode>, Rc<ZVTreeNode>, Rc<ZVTreeNode>)>,
    start: usize,
    end: usize,
}

impl ZVTreeNode {
    fn from(level: u8, high_bits: u64, data: &[u64], start: usize, end: usize) -> ZVTreeNode {
        if level == 32 || end - start <= MAX_ENTRIES_PER_NODE {
            ZVTreeNode {
                children: None,
                start,
                end,
            }
        } else {
            let shift: u8 = (31 - level) * 2;
            let low_bits: u64 = (1_u64 << shift) - 1;
            let bound1 = high_bits;
            let bound2 = high_bits + (0b01_u64 << shift);
            let bound3 = high_bits + (0b10_u64 << shift);
            let bound4 = high_bits + (0b11_u64 << shift);
            let bound5 = high_bits + (0b11_u64 << shift) + low_bits;
            let data_slice = &data[start..end];
            let node1 = Rc::new(ZVTreeNode::from(level + 1, bound1, data, start + data_slice.lower_bound(&bound1), start + data_slice.upper_bound(&(bound2 - 1))));
            let node2 = Rc::new(ZVTreeNode::from(level + 1, bound2, data, start + data_slice.lower_bound(&bound2), start + data_slice.upper_bound(&(bound3 - 1))));
            let node3 = Rc::new(ZVTreeNode::from(level + 1, bound3, data, start + data_slice.lower_bound(&bound3), start + data_slice.upper_bound(&(bound4 - 1))));
            let node4 = Rc::new(ZVTreeNode::from(level + 1, bound4, data, start + data_slice.lower_bound(&bound4), start + data_slice.upper_bound(&bound5)));
            ZVTreeNode {
                children: Some((node1, node2, node3, node4)),
                start,
                end,
            }
        }
    }

    fn size(&self) -> usize {
        16 + if let Some((node1, node2, node3, node4)) = &self.children {
            node1.size() + node2.size() + node3.size() + node4.size() + 32
        } else { 0 }
    }
}

pub struct ZVTree {
    root: ZVTreeNode,
    pub data: Vec<u64>,
}

impl ZVTree {
    pub fn from(input: &[Point]) -> ZVTree {
        let mut data: Vec<u64> = input.iter().map(|p| p.to_zvalue()).collect();
        data.sort_unstable();
        ZVTree {
            root: ZVTreeNode::from(0, 0, &data, 0, data.len()),
            data,
        }
    }
    
    pub fn from_sorted_zvs(input: Vec<u64>) -> ZVTree {
        ZVTree {
            root: ZVTreeNode::from(0, 0, &input, 0, input.len()),
            data: input,
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn size(&self) -> usize {
        self.data.len() * 16 + self.root.size()
    }

    pub fn check_bound(&self, offset: usize, lowx: u32, lowy: u32, highx: u32, highy: u32) -> bool {
        let (x, y) = Point::zvalue_to_raw(self.data[offset]);
        x >= lowx && x <= highx && y >= lowy && y <= highy
    }

    fn range_recursive(&self, node: &ZVTreeNode, lowx: u32, lowy: u32, highx: u32, highy: u32, level: u8, res: &mut Vec<Point>) {
        let curbit_mask: u32 = 1_u32 << (31 - level);
        let lowbit_mask: u32 = curbit_mask - 1;
        let highbit_mask: u32 = !(curbit_mask | lowbit_mask);
        assert_eq!(lowx & highbit_mask, highx & highbit_mask);
        assert_eq!(lowy & highbit_mask, highy & highbit_mask);
        let center_x: u32 = (lowx & highbit_mask) | curbit_mask;
        let center_y: u32 = (lowy & highbit_mask) | curbit_mask;
        if lowx & curbit_mask == 0 && lowx & lowbit_mask == 0 &&
           highx & curbit_mask != 0 && highx & lowbit_mask == lowbit_mask && 
           lowy & curbit_mask == 0 && lowy & lowbit_mask == 0 &&
           highy & curbit_mask != 0 && highy & lowbit_mask == lowbit_mask {

            for i in node.start..node.end {
                res.push(Point::from_zvalue(self.data[i]));
            }
        } else {
            match &node.children {
                None => {
                    for i in node.start..node.end {
                        if self.check_bound(i, lowx, lowy, highx, highy) {
                            res.push(Point::from_zvalue(self.data[i]));
                        }
                    }
                }
                Some((node1, node2, node3, node4)) => {
                    //  o o
                    //  x o
                    if lowx & curbit_mask == 0 && lowy & curbit_mask == 0 {
                        self.range_recursive(&node1, lowx, lowy, (center_x - 1).min(highx) , (center_y - 1).min(highy), level + 1, res);
                    }
                    //  x o
                    //  o o
                    if lowx & curbit_mask == 0 && highy & curbit_mask != 0 {
                        self.range_recursive(&node2, lowx, center_y.max(lowy), (center_x - 1).min(highx), highy, level + 1, res);
                    }
                    // o o
                    // o x
                    if highx & curbit_mask != 0 && lowy & curbit_mask == 0 {
                        self.range_recursive(&node3, center_x.max(lowx), lowy, highx, (center_y - 1).min(highy), level + 1, res);
                    }
                    // o x
                    // o o
                    if highx & curbit_mask != 0 && highy & curbit_mask != 0 {
                        self.range_recursive(&node4, center_x.max(lowx), center_y.max(lowy), highx, highy, level + 1, res);
                    }
                }
            }
        }
    }
    
    fn range_intervals(&self, node: &ZVTreeNode, lowx: u32, lowy: u32, highx: u32, highy: u32, level: u32, intervals: &mut Vec<(usize, usize)>) {
        let curbit_mask: u32 = 1_u32 << (31 - level);
        let lowbit_mask: u32 = curbit_mask - 1;
        let highbit_mask: u32 = !(curbit_mask | lowbit_mask);
        assert_eq!(lowx & highbit_mask, highx & highbit_mask);
        assert_eq!(lowy & highbit_mask, highy & highbit_mask);
        let center_x: u32 = (lowx & highbit_mask) | curbit_mask;
        let center_y: u32 = (lowy & highbit_mask) | curbit_mask;
        if lowx & curbit_mask == 0 && lowx & lowbit_mask == 0 &&
           highx & curbit_mask != 0 && highx & lowbit_mask == lowbit_mask && 
           lowy & curbit_mask == 0 && lowy & lowbit_mask == 0 &&
           highy & curbit_mask != 0 && highy & lowbit_mask == lowbit_mask {

           if node.end - node.start > 0 {
               intervals.push((node.start, node.end));
           }
        } else {
            match &node.children {
                None => {
                    intervals.push((node.start, node.end));
                }
                Some((node1, node2, node3, node4)) => {
                    //  o o
                    //  x o
                    if lowx & curbit_mask == 0 && lowy & curbit_mask == 0 {
                        self.range_intervals(&node1, lowx, lowy, (center_x - 1).min(highx) , (center_y - 1).min(highy), level + 1, intervals);
                    }
                    //  x o
                    //  o o
                    if lowx & curbit_mask == 0 && highy & curbit_mask != 0 {
                        self.range_intervals(&node2, lowx, center_y.max(lowy), (center_x - 1).min(highx), highy, level + 1, intervals);
                    }
                    // o o
                    // o x
                    if highx & curbit_mask != 0 && lowy & curbit_mask == 0 {
                        self.range_intervals(&node3, center_x.max(lowx), lowy, highx, (center_y - 1).min(highy), level + 1, intervals);
                    }
                    // o x
                    // o o
                    if highx & curbit_mask != 0 && highy & curbit_mask != 0 {
                        self.range_intervals(&node4, center_x.max(lowx), center_y.max(lowy), highx, highy, level + 1, intervals);
                    }
                }
            }
        }
    }

    pub fn range(&self, query: &MBR) -> Vec<Point> {
        let mut res: Vec<Point> = Vec::new();
        let (lowx, lowy) = query.low.get_scaled();
        let (highx, highy) = query.high.get_scaled();
        self.range_recursive(&self.root, lowx, lowy, highx, highy, 0, &mut res);
        res
    }

    pub fn get_top_level_alias(&self, query: &MBR) -> (usize, Option<AliasTable>, Vec<(usize, usize)>) {
        let (lowx, lowy) = query.low.get_scaled();
        let (highx, highy) = query.high.get_scaled();
        let mut intervals: Vec<(usize, usize)> = Vec::new();
        self.range_intervals(&self.root, lowx, lowy, highx, highy, 0, &mut intervals);
        let weights: Vec<f64> = intervals.iter().map(|(x, y)| (y - x) as f64).collect();
        let tot_size = weights.iter().sum::<f64>() as usize;
        let alias = AliasTable::from(&weights);
        if intervals.is_empty() {
            (0, None, intervals)
        } else {
            (tot_size, Some(alias), intervals)
        }
    }

    pub fn range_sampling(&self, query: &MBR, k: usize) -> Vec<Point> {
        let (lowx, lowy) = query.low.get_scaled();
        let (highx, highy) = query.high.get_scaled();
        let mut intervals: Vec<(usize, usize)> = Vec::new();
        self.range_intervals(&self.root, lowx, lowy, highx, highy, 0, &mut intervals);

        let weights: Vec<f64> = intervals.iter().map(|(x, y)| (y - x) as f64).collect();
        let dist = Uniform::from(0.0f64..1.0f64);
        let mut rng = util::new_rng();
        let top_level_alias = AliasTable::from(&weights);
        let mut samples: Vec<Point> = Vec::new();
        // For each sample, construct two level sampling.
        while samples.len() < k {
            let coin1 = dist.sample(&mut rng);
            let coin2 = dist.sample(&mut rng);
            let res = top_level_alias.sample(coin1, coin2);
            let coin3 = dist.sample(&mut rng);
            let offset = (weights[res] as f64 * coin3) as usize  + intervals[res].0;
            if self.check_bound(offset, lowx, lowy, highx, highy) {
                samples.push(Point::from_zvalue(self.data[offset]));
            }
        }
        samples
    }

}