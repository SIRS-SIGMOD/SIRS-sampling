use crate::geo::{MBR, Point};
use crate::alias::AliasTable;
use crate::util;
use std::rc::Rc;
use rand::distributions::{Uniform, Distribution};

const KDTREE_THRESHOLD: usize = 256;
// Should maintain balance up to roughly 1/20
// const KDTREE_MED_SAMPLE: usize = 200;

struct KDTreeNode {
    bounding_box: MBR,
    children: Option<(Rc<KDTreeNode>, Rc<KDTreeNode>)>,
    start: usize,
    end: usize,
}

pub struct KDTree {
    root: KDTreeNode,
    data: Vec<Point>,
}

impl KDTreeNode {
    fn new(points: &mut[Point], level: usize, start: usize, end: usize, bounding_box: MBR) -> KDTreeNode {
        assert_eq!(end - start, points.len());
        let len = points.len();
        if len < KDTREE_THRESHOLD {
            KDTreeNode {
                bounding_box: bounding_box,
                children: None,
                start,
                end,
            }
        } else {
            let mid = len / 2;
            let mut left_bounding_box = bounding_box.clone();
            let mut right_bounding_box = bounding_box.clone();
            if level % 3 == 0 {
                let split = order_stat::kth_by(points, mid, |p1, p2| p1.x.partial_cmp(&p2.x).unwrap());
                left_bounding_box.high.x = split.x;
                right_bounding_box.low.x = split.x;
            } else if level % 3 == 1{
                let split = order_stat::kth_by(points, mid, |p1, p2| p1.y.partial_cmp(&p2.y).unwrap());
                left_bounding_box.high.y = split.y;
                right_bounding_box.low.y = split.y;
            } else {
                let split = order_stat::kth_by(points, mid, |p1, p2| p1.z.partial_cmp(&p2.z).unwrap());
                left_bounding_box.high.z = split.z;
                right_bounding_box.low.z = split.z;
            }
            let left_node = KDTreeNode::new(&mut points[0..mid], level + 1, start, start + mid, left_bounding_box);
            let right_node = KDTreeNode::new(&mut points[mid..len], level + 1, start + mid, end, right_bounding_box);
            KDTreeNode {
                bounding_box: bounding_box,
                children: Some((Rc::new(left_node), Rc::new(right_node))),
                start,
                end,
            }
        }
    }

    fn size(&self) -> usize {
        64 + if let Some((left, right)) = &self.children {
            16 + left.size() + right.size()
        } else { 0 }
    }
}

impl KDTree {
    pub fn from(data: &[Point]) -> KDTree {
        let mut points_data: Vec<Point> = Vec::new();
        points_data.extend_from_slice(&data);
        let root = KDTreeNode::new(&mut points_data, 0, 0, data.len(), MBR::from_points(data));
        KDTree {
            root,
            data: points_data,
        }
    }

    pub fn size(&self) -> usize {
        &self.data.len() * 24 + self.root.size()
    }

    pub fn range(&self, query: &MBR) -> Vec<Point> {
        let mut res: Vec<Point> = Vec::new();
        let mut stack: Vec<&KDTreeNode> = Vec::new();
        if query.intersects(&self.root.bounding_box) { stack.push(&self.root); }
        while !stack.is_empty() {
            let now = stack.pop().unwrap();
            match &now.children {
                None => {
                    for i in now.start..now.end {
                        if query.contains(&self.data[i]) {
                            res.push(self.data[i].clone());
                        }
                    }
                }
                Some((left, right)) => {
                    if query.intersects(&left.bounding_box) { stack.push(left); }
                    if query.intersects(&right.bounding_box) { stack.push(right); }
                }
            }
        }
        res
    }

    pub fn olken_range_sampling(&self, query: &MBR, k: usize) -> Vec<Point> {
        let mut samples: Vec<Point> = Vec::new();
        if !query.intersects(&self.root.bounding_box) {
            return samples;
        }
        let mut lca_root: &KDTreeNode = &self.root;
        loop {
            match &lca_root.children {
                None => { break; }
                Some((left, right)) => {
                    let mut cnt = 0;
                    let mut tmp = lca_root;
                    if query.intersects(&left.bounding_box) {
                        cnt += 1;
                        tmp = &left;
                    }
                    if query.intersects(&right.bounding_box) {
                        cnt += 1;
                        tmp = &right;
                    }
                    if cnt == 1 { lca_root = tmp; }
                    else { break; }
                }
            }
        }

        let mut rng = util::new_rng();
        let dist = Uniform::from(0.0f64..1.0f64);
        while samples.len() < k {
            let mut now: &KDTreeNode = &self.root;
            loop {
                match &now.children {
                    None => {
                        let offset = (dist.sample(&mut rng) * (now.end - now.start) as f64) as usize + now.start;
                        if query.contains(&self.data[offset]) {
                            samples.push(self.data[offset].clone());
                        }
                        break;
                    }
                    Some((left, right)) => {
                        let left_count = left.end - left.start;
                        let right_count = right.end - right.start;
                        let p: f64 = left_count as f64 / (left_count + right_count) as f64;
                        if dist.sample(&mut rng) < p { now = &left; }
                        else { now = &right; }
                        //Rejection
                        if !query.intersects(&now.bounding_box) {
                            break;
                        }
                    }
                }
            }
        }
        samples
    }

    pub fn range_sampling(&self, query: &MBR, k: usize) -> Vec<Point> {
        let mut samples: Vec<Point> = Vec::new();
        let mut candidates: Vec<&KDTreeNode> = Vec::new();
        //let mut spare: Vec<usize> = Vec::new();
        let mut stack: Vec<&KDTreeNode> = Vec::new();
        if query.intersects(&self.root.bounding_box) { stack.push(&self.root); }
        while !stack.is_empty() {
            let now = stack.pop().unwrap();
            if query.contains_mbr(&now.bounding_box) {
                candidates.push(now);
            } else {
                match &now.children {
                    None => {
                        candidates.push(now);
                        /*
                        for i in now.start..now.end {
                            if query.contains(&self.data[i]) {
                                spare.push(i);
                            }
                        }*/
                    }
                    Some((left, right)) => {
                        if query.intersects(&left.bounding_box) { stack.push(left); }
                        if query.intersects(&right.bounding_box) { stack.push(right); }
                    }
                }
            }
        }
        // Construct top level alias structure
        //let mut weights = vec![spare.len() as f64];
        let weights: Vec<f64> = candidates.iter().map(|node| (node.end - node.start) as f64).collect();
        let dist = Uniform::from(0.0f64..1.0f64);
        let mut rng = util::new_rng();
        let top_level_alias = AliasTable::from(&weights);

        // For each sample, construct two level sampling.
        while samples.len() < k {
            let coin1 = dist.sample(&mut rng);
            let coin2 = dist.sample(&mut rng);
            let res = top_level_alias.sample(coin1, coin2);
            let coin3 = dist.sample(&mut rng);
            let offset = (weights[res] * coin3) as usize  + candidates[res].start;
            if query.contains(&self.data[offset]) {
                samples.push(self.data[offset].clone());
            }
        }

        samples
    }

}