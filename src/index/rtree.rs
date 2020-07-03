use crate::geo::{MBR, Point};
use crate::alias::AliasTable;
use crate::util;
use std::rc::Rc;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use rand::distributions::{Uniform, Distribution};

struct RTreeNode {
    bounding_box: MBR,
    size: usize,
    children: Option<Vec<Rc<RTreeNode>>>,
    offsets: Vec<usize>,
    alias: AliasTable,
}

impl RTreeNode {
    fn from_data(data: &[Point], offset: usize) -> RTreeNode {
        let mut minx = std::f64::MAX;
        let mut miny = std::f64::MAX;
        let mut maxx = std::f64::MIN;
        let mut maxy = std::f64::MIN;
        let mut offsets: Vec<usize> = Vec::new();
        for (i, p) in data.iter().enumerate() {
            minx = minx.min(p.x);
            miny = miny.min(p.y);
            maxx = maxx.max(p.x);
            maxy = maxy.max(p.y);
            offsets.push(offset + i);
        }

        RTreeNode {
            bounding_box: MBR {
                low: Point {
                    x: minx,
                    y: miny,
                },
                high: Point {
                    x: maxx,
                    y: maxy,
                }
            },
            size: data.len(),
            children: None,
            offsets,
            alias: AliasTable::uniform(data.len()),
        }
    }

    fn from_nodes(nodes: &[Rc<RTreeNode>]) -> RTreeNode {
        let mut minx = std::f64::MAX;
        let mut miny = std::f64::MAX;
        let mut maxx = std::f64::MIN;
        let mut maxy = std::f64::MIN;
        let mut size = 0_usize;
        let mut children: Vec<Rc<RTreeNode>> = Vec::new();
        let mut sizes: Vec<f64> = Vec::new();
        let mut offsets: Vec<usize> = Vec::new();
        for node in nodes.iter() {
            minx = minx.min(node.bounding_box.low.x);
            miny = miny.min(node.bounding_box.low.y);
            maxx = maxx.max(node.bounding_box.high.x);
            maxy = maxy.max(node.bounding_box.high.y);
            children.push(node.clone());
            size += node.size;
            sizes.push(node.size as f64);
            offsets.extend_from_slice(node.offsets.as_slice());
        }
        assert_eq!(size, offsets.len());

        RTreeNode {
            bounding_box: MBR {
                low: Point {
                    x: minx,
                    y: miny,
                },
                high: Point {
                    x: maxx,
                    y: maxy,
                }
            },
            size,
            children: Some(children),
            offsets,
            alias: AliasTable::uniform(size),
        }
    }
}

pub struct RTree {
    root: Rc<RTreeNode>,
    data: Vec<Point>,
}

const MAX_ENTRIES_PER_NODE: usize = 25;

enum KNNEntry {
    Leaf(Point),
    Node(Rc<RTreeNode>),
}

struct KNNUtil {
    dist: f64,
    payload: KNNEntry,
}

impl Ord for KNNUtil {
    fn cmp(&self, other:&Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl PartialOrd for KNNUtil {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.dist.partial_cmp(&self.dist)
    }
}

impl PartialEq for KNNUtil {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist
    }
}

impl Eq for KNNUtil {}

impl RTree {
    pub fn from(data: &[Point]) -> RTree {
        let mut points: Vec<Point> = Vec::new();
        points.extend_from_slice(data);
        let mut now = (points.len() as f64 / MAX_ENTRIES_PER_NODE as f64).ceil() as usize;
        let length = points.len();
        let now_x = (now as f64).sqrt().ceil() as usize;
        let now_y = (now as f64 / now_x as f64).ceil() as usize;
        points.sort_unstable_by(|p1, p2| p1.x.partial_cmp(&p2.x).unwrap());
        let step_x = (points.len() as f64 / now_x as f64).ceil() as usize;
        let mut i = 0 as usize;
        let mut rtree_nodes: Vec<Rc<RTreeNode>> = Vec::new();
        while i < length {
            let data = &mut points[i..(i + step_x).min(length)];
            data.sort_unstable_by(|p1, p2| p1.y.partial_cmp(&p2.y).unwrap());
            let step_y = (std::cmp::min(step_x, length - i) as f64 / now_y as f64).ceil() as usize;
            let mut j = 0 as usize;
            while j < data.len() {
                let len = step_y.min(data.len() - j);
                rtree_nodes.push(Rc::new(RTreeNode::from_data(&data[j..j+len], i + j)));
                j += len;
            }
            i += data.len();
        }

        now = (rtree_nodes.len() as f64 / MAX_ENTRIES_PER_NODE as f64).ceil() as usize;
        while now > 1 {
            let now_x = (now as f64).ceil().sqrt() as usize;
            let now_y = (now as f64 / now_x as f64).ceil() as usize;
            rtree_nodes.sort_unstable_by(
                |n1, n2| (n1.bounding_box.low.x + n1.bounding_box.high.x)
                            .partial_cmp(&(n2.bounding_box.low.x + n2.bounding_box.high.x)).unwrap());
            let length = rtree_nodes.len(); 
            let step_x = (rtree_nodes.len() as f64 / now_x as f64).ceil() as usize;
            let mut i = 0 as usize;
            let mut tmp_nodes: Vec<Rc<RTreeNode>> = Vec::new();
            while i < length {
                let data = &mut rtree_nodes[i..(i + step_x).min(length)];
                data.sort_unstable_by(
                    |n1, n2| (n1.bounding_box.low.y + n1.bounding_box.high.y)
                                .partial_cmp(&(n2.bounding_box.low.y + n2.bounding_box.high.y)).unwrap());
                let step_y = (std::cmp::min(step_x, length - i) as f64 / now_y as f64).ceil() as usize;
                let mut j = 0 as usize;
                while j < data.len() {
                    let len = step_y.min(data.len() - j);
                    tmp_nodes.push(Rc::new(RTreeNode::from_nodes(&data[j..j+len])));
                    j += len;
                }
                i += data.len();
            }
            rtree_nodes = tmp_nodes;
            now = (rtree_nodes.len() as f64 / MAX_ENTRIES_PER_NODE as f64) as usize;
        }

        RTree {
            root: Rc::new(RTreeNode::from_nodes(rtree_nodes.as_slice())),
            data: points,
        }
    }

    pub fn range(&self, query: &MBR) -> Vec<Point> {
        let mut res: Vec<Point> = Vec::new();
        let mut stack: Vec<&RTreeNode> = Vec::new();
        stack.push(&self.root);
        while !stack.is_empty() {
            let now = stack.pop().unwrap();
            match &now.children {
                Some(children) => {
                    for child in children.iter() {
                        if query.intersects(&child.bounding_box) {
                            stack.push(&child);
                        }
                    }
                }
                None => {
                    for offset in now.offsets.iter() {
                        if query.contains(&self.data[*offset]) {
                            res.push(self.data[*offset].clone());
                        }
                    }
                }
            }
        }
        res
    }

    pub fn knn(&self, query: &Point, k: usize) -> Vec<Point> {
        let mut res: Vec<Point> = Vec::new();
        let mut pq: BinaryHeap<KNNUtil> = BinaryHeap::new();
        pq.push(KNNUtil{
            dist: query.min_dist_mbr(&self.root.bounding_box),
            payload: KNNEntry::Node(self.root.clone()),
        });
        while !pq.is_empty() {
            let now = pq.pop().unwrap();
            match now.payload {
                KNNEntry::Node(node) => {
                    if let Some(children) = &node.children {
                        for child in children.iter() {
                            pq.push(KNNUtil {
                                dist: query.min_dist_mbr(&child.bounding_box),
                                payload: KNNEntry::Node(child.clone()),
                            })
                        }
                    } else {
                        for offset in node.offsets.iter() {
                            pq.push(KNNUtil{
                                dist: query.min_dist(&self.data[*offset]),
                                payload: KNNEntry::Leaf(self.data[*offset].clone()),
                            })
                        }
                    }
                }
                KNNEntry::Leaf(p) => {
                    res.push(p.clone());
                    if res.len() == k { break; }
                }
            }
        }
        res
    }

    pub fn range_sampling(&self, query: &MBR, k: usize) -> Vec<Point> {
        let mut samples: Vec<Point> = Vec::new();

        // First, we found all candidate alias tables and construct a
        // spare alias table for spilled points.
        let mut candidates: Vec<&RTreeNode> = Vec::new();
        let mut spare: Vec<usize> = Vec::new();
        let mut stack: Vec<&RTreeNode> = Vec::new();
        stack.push(&self.root);
        while !stack.is_empty() {
            let now = stack.pop().unwrap();
            if query.contains_mbr(&now.bounding_box) {
                candidates.push(&now);
            } else {
                match &now.children {
                    Some(children) => {
                        for child in children.iter() {
                            if query.intersects(&child.bounding_box) {
                                stack.push(child);
                            }
                        }
                    }
                    None => {
                        for offset in now.offsets.iter() {
                            if query.contains(&self.data[*offset]) {
                                spare.push(*offset);
                            }
                        }
                    }
                }
            }
        }
        let spare_alias = AliasTable::uniform(spare.len());
        
        // Construct top level alias structure
        let mut weights = vec![spare.len() as f64];
        weights.extend(candidates.iter().map(|node| node.size as f64));
        let dist = Uniform::from(0.0f64..1.0f64);
        let mut rng = util::new_rng();
        let top_level_alias = AliasTable::from(&weights);

        // For each sample, construct two level sampling.
        for _ in 0..k {
            let coin1 = dist.sample(&mut rng);
            let coin2 = dist.sample(&mut rng);
            let res = top_level_alias.sample(coin1, coin2);
            let current_table = if res == 0 { &spare_alias } else { &candidates[res - 1].alias };
            let offset_table = if res == 0 { &spare } else { &candidates[res - 1].offsets };
            let coin3 = dist.sample(&mut rng);
            let coin4 = dist.sample(&mut rng);
            samples.push(self.data[offset_table[current_table.sample(coin3, coin4)]].clone());
        }

        samples
    }
}
