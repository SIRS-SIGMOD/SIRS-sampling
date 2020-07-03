use crate::alias::AliasTable;
use crate::geo::{MBR, Point};
use crate::util;
use superslice::*;
use std::rc::Rc;
use rand::distributions::{Uniform, Distribution};

const MAX_ENTRIES_PER_NODE: usize = 256;

struct ZVTreeNode {
    children: Option<(Rc<ZVTreeNode>, Rc<ZVTreeNode>, Rc<ZVTreeNode>, Rc<ZVTreeNode>)>,
    alias: AliasTable,
    weight: f64,
    start: usize,
    end: usize,
}

impl ZVTreeNode {
    fn from(level: u8, high_bits: u64, data: &[(u64, f64)], start: usize, end: usize, weight: &mut f64) -> ZVTreeNode {
        if level == 32 || end - start <= MAX_ENTRIES_PER_NODE {
            let weights: Vec<f64> = data[start..end].iter().map(|x| x.1).collect();
            *weight = weights.iter().sum();
            ZVTreeNode {
                children: None,
                alias: AliasTable::from(&weights),
                weight: *weight,
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
            let mut weight1 = 0.0; let mut weight2 = 0.0; let mut weight3 = 0.0; let mut weight4 = 0.0;
            let node1 = Rc::new(ZVTreeNode::from(level + 1, bound1, data, start + data_slice.lower_bound_by_key(&bound1, |i| i.0), start + data_slice.upper_bound_by_key(&(bound2 - 1), |i| i.0), &mut weight1));
            let node2 = Rc::new(ZVTreeNode::from(level + 1, bound2, data, start + data_slice.lower_bound_by_key(&bound2, |i| i.0), start + data_slice.upper_bound_by_key(&(bound3 - 1), |i| i.0), &mut weight2));
            let node3 = Rc::new(ZVTreeNode::from(level + 1, bound3, data, start + data_slice.lower_bound_by_key(&bound3, |i| i.0), start + data_slice.upper_bound_by_key(&(bound4 - 1), |i| i.0), &mut weight3));
            let node4 = Rc::new(ZVTreeNode::from(level + 1, bound4, data, start + data_slice.lower_bound_by_key(&bound4, |i| i.0), start + data_slice.upper_bound_by_key(&bound5, |i| i.0), &mut weight4));
            *weight = weight1 + weight2 + weight3 + weight4;
            ZVTreeNode {
                children: Some((node1, node2, node3, node4)),
                alias: AliasTable::from(&vec![weight1, weight2, weight3, weight4]),
                weight: *weight,
                start,
                end,
            }
        }
    }

    fn size(&self) -> usize {
        24 + if let Some((node1, node2, node3, node4)) = &self.children {
            node1.size() + node2.size() + node3.size() + node4.size() + 32 + 64
        } else { (self.end - self.start) * 16 }
    }
}

pub struct ZVTree {
    root: ZVTreeNode,
    data: Vec<(u64, f64)>,
}

impl ZVTree {
    pub fn from_zvpoints(mut data: Vec<(u64, f64)>) -> ZVTree {
        data.sort_unstable_by_key(|p| p.0);
        ZVTree {
            root: ZVTreeNode::from(0, 0, &data, 0, data.len(), &mut 0.0_f64),
            data,
        }
    }

    pub fn from(input: &[util::WPoint]) -> ZVTree {
        let mut data: Vec<(u64, f64)> = input.iter().map(|item| (item.p.to_zvalue(), item.weight)).collect();
        data.sort_unstable_by_key(|p| p.0);
        ZVTree {
            root: ZVTreeNode::from(0, 0, &data, 0, data.len(), &mut 0.0_f64),
            data,
        }
    }

    pub fn size(&self) -> usize {
        self.data.len() * 16 + self.root.size()
    }

    fn check_bound(&self, offset: usize, lowx: u32, lowy: u32, highx: u32, highy: u32) -> bool {
        let (x, y) = Point::zvalue_to_raw(self.data[offset].0);
        x >= lowx && x <= highx && y >= lowy && y <= highy
    }

    fn range_recursive(&self, node: &ZVTreeNode, lowx: u32, lowy: u32, highx: u32, highy: u32, level: u8, res: &mut Vec<util::WPoint>) {
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
                res.push(util::WPoint{p: Point::from_zvalue(self.data[i].0), weight: self.data[i].1});
            }
        } else {
            match &node.children {
                None => {
                    for i in node.start..node.end {
                        if self.check_bound(i, lowx, lowy, highx, highy) {
                            res.push(util::WPoint{p: Point::from_zvalue(self.data[i].0), weight: self.data[i].1});
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
    
    fn range_intervals(&self, node: &ZVTreeNode, lowx: u32, lowy: u32, highx: u32, highy: u32, level: u32, candidates: &mut Vec<*const ZVTreeNode>) {
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
               candidates.push(node as *const ZVTreeNode);
           }
        } else {
            match &node.children {
                None => {
                    candidates.push(node as *const ZVTreeNode);
                }
                Some((node1, node2, node3, node4)) => {
                    //  o o
                    //  x o
                    if lowx & curbit_mask == 0 && lowy & curbit_mask == 0 {
                        self.range_intervals(&node1, lowx, lowy, (center_x - 1).min(highx) , (center_y - 1).min(highy), level + 1, candidates);
                    }
                    //  x o
                    //  o o
                    if lowx & curbit_mask == 0 && highy & curbit_mask != 0 {
                        self.range_intervals(&node2, lowx, center_y.max(lowy), (center_x - 1).min(highx), highy, level + 1, candidates);
                    }
                    // o o
                    // o x
                    if highx & curbit_mask != 0 && lowy & curbit_mask == 0 {
                        self.range_intervals(&node3, center_x.max(lowx), lowy, highx, (center_y - 1).min(highy), level + 1, candidates);
                    }
                    // o x
                    // o o
                    if highx & curbit_mask != 0 && highy & curbit_mask != 0 {
                        self.range_intervals(&node4, center_x.max(lowx), center_y.max(lowy), highx, highy, level + 1, candidates);
                    }
                }
            }
        }
    }

    pub fn range(&self, query: &MBR) -> Vec<util::WPoint> {
        let mut res: Vec<util::WPoint> = Vec::new();
        let (lowx, lowy) = query.low.get_scaled();
        let (highx, highy) = query.high.get_scaled();
        self.range_recursive(&self.root, lowx, lowy, highx, highy, 0, &mut res);
        res
    }

    fn sample_subtree(node: &ZVTreeNode, rng: &mut impl rand::RngCore, dist: &Uniform<f64>) -> usize {
        let mut now: &ZVTreeNode = node;
        loop {
            let coin1 = dist.sample(rng);
            let coin2 = dist.sample(rng);
            match &now.children {
                None => {
                    return now.start + now.alias.sample(coin1, coin2)
                }
                Some((node1, node2, node3, node4)) => {
                    match now.alias.sample(coin1, coin2) {
                        0 => now = node1,
                        1 => now = node2,
                        2 => now = node3,
                        3 => now = node4,
                        _ => unreachable!(),
                    }
                }
            }
        }
    }

    pub fn range_sampling(&self, query: &MBR, k: usize) -> Vec<Point> {
        let (lowx, lowy) = query.low.get_scaled();
        let (highx, highy) = query.high.get_scaled();
        let mut candidates: Vec<*const ZVTreeNode> = Vec::new();
        self.range_intervals(&self.root, lowx, lowy, highx, highy, 0, &mut candidates);

        let weights: Vec<f64> = candidates.iter().map(|node| unsafe { (**node).weight }).collect();
        let dist = Uniform::from(0.0f64..1.0f64);
        let mut rng = util::new_rng();
        let top_level_alias = AliasTable::from(&weights);
        let mut samples: Vec<Point> = Vec::new();
        // For each sample, construct two level sampling.
        while samples.len() < k {
            let coin1 = dist.sample(&mut rng);
            let coin2 = dist.sample(&mut rng);
            let res = top_level_alias.sample(coin1, coin2);
            let offset = ZVTree::sample_subtree(unsafe { &*candidates[res] }, &mut rng, &dist);
            if self.check_bound(offset, lowx, lowy, highx, highy) {
                samples.push(Point::from_zvalue(self.data[offset].0));
            }
        }
        samples
    }

}
