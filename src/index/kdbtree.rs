use crate::geo::{MBR, Point};
use crate::util;
use crate::alias::AliasTable;
use std::rc::Rc;
use rand::distributions::{Uniform, Distribution};

const KDB_SAMPLE_BUFFER_SIZE: usize = 128;
const KDBTREE_THRESHOLD: usize = 2 * KDB_SAMPLE_BUFFER_SIZE;

struct KDBTreeNode {
    bounding_box: MBR,
    children: Option<(Rc<KDBTreeNode>, Rc<KDBTreeNode>)>,
    start: usize,
    end: usize,
    sample_buffer: Vec<Point>,
    valid_ptr: usize,
}

pub struct KDBTree {
    root: KDBTreeNode,
    data: Vec<Point>,
}

impl KDBTreeNode {
    fn new(points: &mut[Point], level: usize, start: usize, end: usize, bounding_box: MBR) -> KDBTreeNode {
        assert_eq!(end - start, points.len());
        let len = points.len();
        if len < KDBTREE_THRESHOLD {
            KDBTreeNode {
                bounding_box: bounding_box,
                children: None,
                start,
                end,
                sample_buffer: Vec::new(),
                valid_ptr: 0,
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
            let left_node = KDBTreeNode::new(&mut points[0..mid], level + 1, start, start + mid, left_bounding_box);
            let right_node = KDBTreeNode::new(&mut points[mid..len], level + 1, start + mid, end, right_bounding_box);
            KDBTreeNode {
                bounding_box: bounding_box,
                children: Some((Rc::new(left_node), Rc::new(right_node))),
                start,
                end,
                sample_buffer: util::sample_from(points, KDB_SAMPLE_BUFFER_SIZE),
                valid_ptr: 0,
            }
        }
    }

    fn size(&self) -> usize {
        72 + self.sample_buffer.len() * 24 + if let Some((left, right)) = &self.children {
            16 + left.size() + right.size()
        } else { 0 }
    }
}

impl KDBTree {
    pub fn from(data: &[Point]) -> KDBTree {
        let mut points_data: Vec<Point> = Vec::new();
        points_data.extend_from_slice(data);
        let root = KDBTreeNode::new(&mut points_data, 0, 0, data.len(), MBR::from_points(data));
        KDBTree {
            root,
            data: points_data,
        }
    }

    pub fn size(&self) -> usize {
        self.data.len() * 24 + self.root.size()
    }

    pub fn range(&self, query: &MBR) -> Vec<Point> {
        let mut res: Vec<Point> = Vec::new();
        let mut stack: Vec<&KDBTreeNode> = Vec::new();
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

    pub fn range_sampling(&mut self, query: &MBR, k: usize) -> Vec<Point> {
        let mut samples: Vec<Point> = Vec::new();
        let mut frontier: Vec<*mut KDBTreeNode> = Vec::new();
        let mut new_frontier: Vec<*mut KDBTreeNode> = Vec::new();
        let mut weights: Vec<f64> = Vec::new();
        frontier.push(&mut self.root as *mut KDBTreeNode);
        let mut alias = AliasTable::uniform(1);
        let mut rng = util::new_rng();
        let dist = Uniform::from(0.0f64..1.0f64);
        while samples.len() < k {
            let coin1 = dist.sample(&mut rng);
            let coin2 = dist.sample(&mut rng);
            let offset = alias.sample(coin1, coin2);
            let node = unsafe { &mut *frontier[offset] };
            let mut flag = false;
            match &mut node.children {
                None => {
                    let coin = dist.sample(&mut rng);
                    let sample = &self.data[node.start + ((node.end - node.start) as f64 * coin) as usize];
                    if query.contains(sample) { samples.push(sample.clone()); }
                }
                Some((left, right)) => {
                    assert!(node.valid_ptr < node.sample_buffer.len());
                    let sample = &node.sample_buffer[node.valid_ptr];
                    node.valid_ptr += 1;
                    if query.contains(sample) { samples.push(sample.clone()); }
                    if node.valid_ptr == node.sample_buffer.len() {
                        //rebuild frontier
                        flag = true;
                        weights.clear();
                        new_frontier.clear();
                        for (i, item) in frontier.iter().enumerate() {
                            if i == offset {
                                let left_ptr = (&**left as *const KDBTreeNode) as *mut KDBTreeNode;
                                let right_ptr = (&**right as *const KDBTreeNode) as *mut KDBTreeNode;
                                if left.bounding_box.intersects(&query) {
                                    new_frontier.push(left_ptr);
                                    weights.push((left.end - left.start) as f64);
                                }
                                if right.bounding_box.intersects(&query) {
                                    new_frontier.push(right_ptr);
                                    weights.push((right.end - right.start) as f64);
                                }
                            } else {
                                new_frontier.push(item.clone());
                                weights.push(unsafe { (**item).end - (**item).start } as f64);
                            }
                        }
                        //replenish buffer
                        node.sample_buffer = util::sample_from(&self.data[node.start..node.end], KDB_SAMPLE_BUFFER_SIZE);
                        node.valid_ptr = 0;
                    }
                }
            }
            if flag {
                std::mem::swap(&mut frontier, &mut new_frontier);
                alias = AliasTable::from(&weights);
            }
        }
        samples
    }
}