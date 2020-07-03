use crate::geo::{MBR, Point};
use crate::alias::AliasTable;
use crate::util::WPoint;
use crate::util;
use std::rc::Rc;
use rand::distributions::{Uniform, Distribution};

const KDTREE_THRESHOLD: usize = 256;
// Should maintain balance up to roughly 1/20
// const KDTREE_MED_SAMPLE: usize = 200;

struct KDTreeNode {
    bounding_box: MBR,
    children: Option<(Rc<KDTreeNode>, Rc<KDTreeNode>)>,
    weight: f64,
    start: usize,
    end: usize,
    alias: Option<AliasTable>,
}

pub struct KDTree {
    root: KDTreeNode,
    data: Vec<WPoint>,
}

impl KDTreeNode {
    fn new(points: &mut[WPoint], level: usize, start: usize, end: usize, bounding_box: MBR, weight: &mut f64) -> KDTreeNode {
        assert_eq!(end - start, points.len());
        let len = points.len();
        if len < KDTREE_THRESHOLD {
            let weights: Vec<f64> = points.iter().map(|x| x.weight).collect();
            *weight = weights.iter().sum();
            KDTreeNode {
                bounding_box: bounding_box,
                children: None,
                weight: *weight,
                start,
                end,
                alias: Some(AliasTable::from(&weights)),
            }
        } else {
            let mid = len / 2;
            let mut left_bounding_box = bounding_box.clone();
            let mut right_bounding_box = bounding_box.clone();
            if level % 2 == 0 {
                let split = order_stat::kth_by(points, mid, |p1, p2| p1.p.x.partial_cmp(&p2.p.x).unwrap());
                left_bounding_box.high.x = split.p.x;
                right_bounding_box.low.x = split.p.x;
            } else {
                let split = order_stat::kth_by(points, mid, |p1, p2| p1.p.y.partial_cmp(&p2.p.y).unwrap());
                left_bounding_box.high.y = split.p.y;
                right_bounding_box.low.y = split.p.y;
            }
            let mut left_weight = 0.0_f64;
            let mut right_weight = 0.0_f64;
            let left_node = KDTreeNode::new(&mut points[0..mid], level + 1, start, start + mid, left_bounding_box, &mut left_weight);
            let right_node = KDTreeNode::new(&mut points[mid..len], level + 1, start + mid, end, right_bounding_box, &mut right_weight);
            *weight = left_weight + right_weight;
            KDTreeNode {
                bounding_box: bounding_box,
                children: Some((Rc::new(left_node), Rc::new(right_node))),
                weight: *weight,
                start,
                end,
                alias: None,
            }
        }
    }

    fn size(&self) -> usize {
        56 + if let Some((left, right)) = &self.children {
            16 + left.size() + right.size()
        } else { (self.end - self.start) * 16 }
    }
}

impl KDTree {
    pub fn from(mut data: Vec<WPoint>) -> KDTree {
        let len = data.len();
        let mbr = util::wpoints_to_mbr(&data);
        let mut tot_weight = 0.0_f64;
        let root = KDTreeNode::new(&mut data, 0, 0, len, mbr, &mut tot_weight);
        KDTree {
            root,
            data,
        }
    }

    // FIXME: I need to size of ialias.
    pub fn size(&self) -> usize {
        &self.data.len() * 24 + self.root.size()
    }

    pub fn range(&self, query: &MBR) -> Vec<WPoint> {
        let mut res: Vec<WPoint> = Vec::new();
        let mut stack: Vec<&KDTreeNode> = Vec::new();
        if query.intersects(&self.root.bounding_box) { stack.push(&self.root); }
        while !stack.is_empty() {
            let now = stack.pop().unwrap();
            match &now.children {
                None => {
                    for i in now.start..now.end {
                        if query.contains(&self.data[i].p) {
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

    // This need to be fixed.
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
            let mut now: &KDTreeNode = lca_root;
            loop {
                match &now.children {
                    None => {
                        let coin1 = dist.sample(&mut rng);
                        let coin2 = dist.sample(&mut rng);
                        let offset = if let Some(alias) = &now.alias {
                            now.start + alias.sample(coin1, coin2)
                        } else {
                            unreachable!()
                        };
                        if query.contains(&self.data[offset].p) {
                            samples.push(self.data[offset].p.clone());
                        }
                        break;
                    }
                    Some((left, right)) => {
                        let p: f64 = left.weight as f64 / now.weight as f64;
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

    fn sample_subtree(node: &KDTreeNode, rng: &mut impl rand::RngCore, dist: &Uniform<f64>) -> usize {
        let mut now: &KDTreeNode = node;
        loop {
            match &now.children {
                None => {
                    let coin1 = dist.sample(rng);
                    let coin2 = dist.sample(rng);
                    let offset = if let Some(alias) = &now.alias {
                        now.start + alias.sample(coin1, coin2)
                    } else {
                        unreachable!()
                    };
                    return offset;
                }
                Some((left, right)) => {
                    let p: f64 = left.weight as f64 / now.weight as f64;
                    if dist.sample(rng) < p { now = &left; }
                    else { now = &right; }
                }
            }
        }
    }

    pub fn range_sampling(&self, query: &MBR, k: usize) -> Vec<Point> {
        let mut samples: Vec<Point> = Vec::new();
        let mut candidates: Vec<&KDTreeNode> = Vec::new();
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
                    }
                    Some((left, right)) => {
                        if query.intersects(&left.bounding_box) { stack.push(left); }
                        if query.intersects(&right.bounding_box) { stack.push(right); }
                    }
                }
            }
        }
        // Construct top level alias structure
        let weights: Vec<f64> = candidates.iter().map(|node| node.weight).collect();
        let dist = Uniform::from(0.0f64..1.0f64);
        let mut rng = util::new_rng();
        let top_level_alias = AliasTable::from(&weights);

        // For each sample, construct two level sampling.
        while samples.len() < k {
            let coin1 = dist.sample(&mut rng);
            let coin2 = dist.sample(&mut rng);
            let res = top_level_alias.sample(coin1, coin2);
            let offset = KDTree::sample_subtree(candidates[res], &mut rng, &dist);
            if query.contains(&self.data[offset].p) {
                samples.push(self.data[offset].p.clone());
            }
        }

        samples
    }

}
