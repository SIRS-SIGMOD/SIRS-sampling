use crate::alias::AliasTable;
use crate::geo::{MBR, Point};
use crate::util;
use superslice::*;
use std::rc::Rc;
use rand::distributions::{Uniform, Distribution};

const MAX_ENTRIES_PER_NODE: usize = 256;

struct ZVTreeNode {
    //children: Option<(Rc<ZVTreeNode>, Rc<ZVTreeNode>, Rc<ZVTreeNode>, Rc<ZVTreeNode>)>,
    children: Option<[Rc<ZVTreeNode>; 8]>,
    start: usize,
    end: usize,
}

impl ZVTreeNode {
    fn from(level: u8, high_bits: u128, data: &[u128], start: usize, end: usize) -> ZVTreeNode {
        if level == 32 || end - start <= MAX_ENTRIES_PER_NODE {
            ZVTreeNode {
                children: None,
                start,
                end,
            }
        } else {
            let shift: u8 = (31 - level) * 3;
            let low_bits: u128 = (1_u128 << shift) - 1;
            let bound1 = high_bits;
            let bound2 = high_bits + (0b001_u128 << shift);
            let bound3 = high_bits + (0b010_u128 << shift);
            let bound4 = high_bits + (0b011_u128 << shift);
            let bound5 = high_bits + (0b100_u128 << shift);
            let bound6 = high_bits + (0b101_u128 << shift);
            let bound7 = high_bits + (0b110_u128 << shift);
            let bound8 = high_bits + (0b111_u128 << shift);
            let bound9 = high_bits + (0b111_u128 << shift) + low_bits;
            let data_slice = &data[start..end];
            let node1 = Rc::new(ZVTreeNode::from(level + 1, bound1, data, start + data_slice.lower_bound(&bound1), start + data_slice.upper_bound(&(bound2 - 1))));
            let node2 = Rc::new(ZVTreeNode::from(level + 1, bound2, data, start + data_slice.lower_bound(&bound2), start + data_slice.upper_bound(&(bound3 - 1))));
            let node3 = Rc::new(ZVTreeNode::from(level + 1, bound3, data, start + data_slice.lower_bound(&bound3), start + data_slice.upper_bound(&(bound4 - 1))));
            let node4 = Rc::new(ZVTreeNode::from(level + 1, bound4, data, start + data_slice.lower_bound(&bound4), start + data_slice.upper_bound(&(bound5 - 1))));
            let node5 = Rc::new(ZVTreeNode::from(level + 1, bound5, data, start + data_slice.lower_bound(&bound5), start + data_slice.upper_bound(&(bound6 - 1))));
            let node6 = Rc::new(ZVTreeNode::from(level + 1, bound6, data, start + data_slice.lower_bound(&bound6), start + data_slice.upper_bound(&(bound7 - 1))));
            let node7 = Rc::new(ZVTreeNode::from(level + 1, bound7, data, start + data_slice.lower_bound(&bound7), start + data_slice.upper_bound(&(bound8 - 1))));
            let node8 = Rc::new(ZVTreeNode::from(level + 1, bound8, data, start + data_slice.lower_bound(&bound8), start + data_slice.upper_bound(&bound9)));
            ZVTreeNode {
                children: Some([node1, node2, node3, node4, node5, node6, node7, node8]),
                start,
                end,
            }
        }
    }

    fn size(&self) -> usize {
        16 + if let Some(nodes) = &self.children {
            nodes[0].size() + nodes[1].size() + nodes[2].size() + nodes[3].size() +
            nodes[4].size() + nodes[5].size() + nodes[6].size() + nodes[7].size() + 64
        } else { 0 }
    }
}

pub struct ZVTree {
    root: ZVTreeNode,
    data: Vec<u128>,
}

impl ZVTree {
    pub fn from(input: &[Point]) -> ZVTree {
        let mut data: Vec<u128> = input.iter().map(|p| p.to_zvalue()).collect();
        data.sort_unstable();
        ZVTree {
            root: ZVTreeNode::from(0, 0, &data, 0, data.len()),
            data,
        }
    }

    pub fn size(&self) -> usize {
        self.data.len() * 16 + self.root.size()
    }

    fn check_bound(&self, offset: usize, lowx: u32, lowy: u32, lowz: u32, highx: u32, highy: u32, highz: u32) -> bool {
        let (x, y, z) = Point::zvalue_to_raw(self.data[offset]);
        x >= lowx && x <= highx && y >= lowy && y <= highy && z >= lowz && z <= highz
    }

    fn range_recursive(&self, node: &ZVTreeNode, lowx: u32, lowy: u32, lowz: u32, highx: u32, highy: u32, highz: u32, level: u8, res: &mut Vec<Point>) {
        let curbit_mask: u32 = 1_u32 << (31 - level);
        let lowbit_mask: u32 = curbit_mask - 1;
        let highbit_mask: u32 = !(curbit_mask | lowbit_mask);
        assert_eq!(lowx & highbit_mask, highx & highbit_mask);
        assert_eq!(lowy & highbit_mask, highy & highbit_mask);
        assert_eq!(lowz & highbit_mask, highz & highbit_mask);
        let center_x: u32 = (lowx & highbit_mask) | curbit_mask;
        let center_y: u32 = (lowy & highbit_mask) | curbit_mask;
        let center_z: u32 = (lowz & highbit_mask) | curbit_mask;
        if lowx & curbit_mask == 0 && lowx & lowbit_mask == 0 &&
           highx & curbit_mask != 0 && highx & lowbit_mask == lowbit_mask && 
           lowy & curbit_mask == 0 && lowy & lowbit_mask == 0 &&
           highy & curbit_mask != 0 && highy & lowbit_mask == lowbit_mask &&
           lowz & curbit_mask == 0 && lowz & lowbit_mask == 0 &&
           highz & curbit_mask != 0 && highz & lowbit_mask == lowbit_mask {

            for i in node.start..node.end {
                res.push(Point::from_zvalue(self.data[i]));
            }
        } else {
            match &node.children {
                None => {
                    for i in node.start..node.end {
                        if self.check_bound(i, lowx, lowy, lowz, highx, highy, highz) {
                            res.push(Point::from_zvalue(self.data[i]));
                        }
                    }
                }
                Some(nodes) => {
                    // 000
                    if lowx & curbit_mask == 0 && lowy & curbit_mask == 0 && lowz & curbit_mask == 0 {
                        self.range_recursive(&nodes[0], lowx, lowy, lowz, (center_x - 1).min(highx) , (center_y - 1).min(highy), (center_z - 1).min(highz), level + 1, res);
                    }
                    // 001
                    if lowx & curbit_mask == 0 && lowy & curbit_mask == 0 && highz & curbit_mask != 0 {
                        self.range_recursive(&nodes[1], lowx, lowy, center_z.max(lowz), (center_x - 1).min(highx) , (center_y - 1).min(highy), highz, level + 1, res);
                    }
                    // 010
                    if lowx & curbit_mask == 0 && highy & curbit_mask != 0 && lowz & curbit_mask == 0 {
                        self.range_recursive(&nodes[2], lowx, center_y.max(lowy), lowz, (center_x - 1).min(highx) , highy, (center_z - 1).min(highz), level + 1, res);
                    }
                    // 011
                    if lowx & curbit_mask == 0 && highy & curbit_mask != 0 && highz & curbit_mask != 0 {
                        self.range_recursive(&nodes[3], lowx, center_y.max(lowy), center_z.max(lowz), (center_x - 1).min(highx) , highy, highz, level + 1, res);
                    }
                    // 100
                    if highx & curbit_mask != 0 && lowy & curbit_mask == 0 && lowz & curbit_mask == 0 {
                        self.range_recursive(&nodes[4], center_x.max(lowx), lowy, lowz, highx, (center_y - 1).min(highy), (center_z - 1).min(highz), level + 1, res);
                    }
                    // 101
                    if highx & curbit_mask != 0 && lowy & curbit_mask == 0 && highz & curbit_mask != 0 {
                        self.range_recursive(&nodes[5], center_x.max(lowx), lowy, center_z.max(lowz), highx, (center_y - 1).min(highy), highz, level + 1, res);
                    }
                    // 110
                    if highx & curbit_mask != 0 && highy & curbit_mask != 0 && lowz & curbit_mask == 0 {
                        self.range_recursive(&nodes[6], center_x.max(lowx), center_y.max(lowy), lowz, highx, highy, (center_z - 1).min(highz), level + 1, res);
                    }
                    // 111
                    if highx & curbit_mask != 0 && highy & curbit_mask != 0 && highz & curbit_mask != 0 {
                        self.range_recursive(&nodes[7], center_x.max(lowx), center_y.max(lowy), center_z.max(lowz), highx, highy, highz, level + 1, res);
                    }
                }
            }
        }
    }
    
    fn range_intervals(&self, node: &ZVTreeNode, lowx: u32, lowy: u32, lowz: u32, highx: u32, highy: u32, highz: u32, level: u32, intervals: &mut Vec<(usize, usize)>) {
        let curbit_mask: u32 = 1_u32 << (31 - level);
        let lowbit_mask: u32 = curbit_mask - 1;
        let highbit_mask: u32 = !(curbit_mask | lowbit_mask);
        assert_eq!(lowx & highbit_mask, highx & highbit_mask);
        assert_eq!(lowy & highbit_mask, highy & highbit_mask);
        assert_eq!(lowz & highbit_mask, highz & highbit_mask);
        let center_x: u32 = (lowx & highbit_mask) | curbit_mask;
        let center_y: u32 = (lowy & highbit_mask) | curbit_mask;
        let center_z: u32 = (lowz & highbit_mask) | curbit_mask;
        if lowx & curbit_mask == 0 && lowx & lowbit_mask == 0 &&
           highx & curbit_mask != 0 && highx & lowbit_mask == lowbit_mask && 
           lowy & curbit_mask == 0 && lowy & lowbit_mask == 0 &&
           highy & curbit_mask != 0 && highy & lowbit_mask == lowbit_mask &&
           lowz & curbit_mask == 0 && lowz & lowbit_mask == 0 &&
           highz & curbit_mask != 0 && highz & lowbit_mask == lowbit_mask {

           if node.end - node.start > 0 {
               intervals.push((node.start, node.end));
           }
        } else {
            match &node.children {
                None => {
                    intervals.push((node.start, node.end));
                }
                Some(nodes) => {
                    // 000
                    if lowx & curbit_mask == 0 && lowy & curbit_mask == 0 && lowz & curbit_mask == 0 {
                        self.range_intervals(&nodes[0], lowx, lowy, lowz, (center_x - 1).min(highx) , (center_y - 1).min(highy), (center_z - 1).min(highz), level + 1, intervals);
                    }
                    // 001
                    if lowx & curbit_mask == 0 && lowy & curbit_mask == 0 && highz & curbit_mask != 0 {
                        self.range_intervals(&nodes[1], lowx, lowy, center_z.max(lowz), (center_x - 1).min(highx) , (center_y - 1).min(highy), highz, level + 1, intervals);
                    }
                    // 010
                    if lowx & curbit_mask == 0 && highy & curbit_mask != 0 && lowz & curbit_mask == 0 {
                        self.range_intervals(&nodes[2], lowx, center_y.max(lowy), lowz, (center_x - 1).min(highx) , highy, (center_z - 1).min(highz), level + 1, intervals);
                    }
                    // 011
                    if lowx & curbit_mask == 0 && highy & curbit_mask != 0 && highz & curbit_mask != 0 {
                        self.range_intervals(&nodes[3], lowx, center_y.max(lowy), center_z.max(lowz), (center_x - 1).min(highx) , highy, highz, level + 1, intervals);
                    }
                    // 100
                    if highx & curbit_mask != 0 && lowy & curbit_mask == 0 && lowz & curbit_mask == 0 {
                        self.range_intervals(&nodes[4], center_x.max(lowx), lowy, lowz, highx, (center_y - 1).min(highy), (center_z - 1).min(highz), level + 1, intervals);
                    }
                    // 101
                    if highx & curbit_mask != 0 && lowy & curbit_mask == 0 && highz & curbit_mask != 0 {
                        self.range_intervals(&nodes[5], center_x.max(lowx), lowy, center_z.max(lowz), highx, (center_y - 1).min(highy), highz, level + 1, intervals);
                    }
                    // 110
                    if highx & curbit_mask != 0 && highy & curbit_mask != 0 && lowz & curbit_mask == 0 {
                        self.range_intervals(&nodes[6], center_x.max(lowx), center_y.max(lowy), lowz, highx, highy, (center_z - 1).min(highz), level + 1, intervals);
                    }
                    // 111
                    if highx & curbit_mask != 0 && highy & curbit_mask != 0 && highz & curbit_mask != 0 {
                        self.range_intervals(&nodes[7], center_x.max(lowx), center_y.max(lowy), center_z.max(lowz), highx, highy, highz, level + 1, intervals);
                    }
                }
            }
        }
    }

    pub fn range(&self, query: &MBR) -> Vec<Point> {
        let mut res: Vec<Point> = Vec::new();
        let (lowx, lowy, lowz) = query.low.get_scaled();
        let (highx, highy, highz) = query.high.get_scaled();
        self.range_recursive(&self.root, lowx, lowy, lowz, highx, highy, highz, 0, &mut res);
        res
    }

    pub fn range_sampling(&self, query: &MBR, k: usize) -> Vec<Point> {
        let (lowx, lowy, lowz) = query.low.get_scaled();
        let (highx, highy, highz) = query.high.get_scaled();
        let mut intervals: Vec<(usize, usize)> = Vec::new();
        self.range_intervals(&self.root, lowx, lowy, lowz, highx, highy, highz, 0, &mut intervals);

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
            if self.check_bound(offset, lowx, lowy, lowz, highx, highy, highz) {
                samples.push(Point::from_zvalue(self.data[offset]));
            }
        }
        samples
    }

}
