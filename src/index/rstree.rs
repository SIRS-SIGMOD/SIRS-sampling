use crate::geo::{MBR, Point};
use crate::alias::AliasTable;
use crate::util;
use std::rc::Rc;
use rand::distributions::{Uniform, Distribution};

struct RSTreeNode {
    bounding_box: MBR,
    size: usize,
    children: Option<Vec<Rc<RSTreeNode>>>,
    offset: usize,
}

impl RSTreeNode {
    fn size(&self) -> usize {
        48 + if let Some(children) = &self.children {
            let mut res = children.len() * 8;
            for child in children.iter() {
                res += child.size()
            }
            res
        } else { 0 }
    }

    fn from_data(data: &[Point], offset: usize) -> RSTreeNode {
        let mut minx = std::f64::MAX;
        let mut miny = std::f64::MAX;
        let mut maxx = std::f64::MIN;
        let mut maxy = std::f64::MIN;
        for p in data.iter() {
            minx = minx.min(p.x); miny = miny.min(p.y);
            maxx = maxx.max(p.x); maxy = maxy.max(p.y);
        }

        RSTreeNode {
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
            offset,
        }
    }

    fn from_nodes(nodes: &[Rc<RSTreeNode>]) -> RSTreeNode {
        let mut minx = std::f64::MAX;
        let mut miny = std::f64::MAX;
        let mut maxx = std::f64::MIN;
        let mut maxy = std::f64::MIN;
        let mut size = 0_usize;
        let mut children: Vec<Rc<RSTreeNode>> = Vec::new();
        let mut sizes: Vec<f64> = Vec::new();
        for node in nodes.iter() {
            minx = minx.min(node.bounding_box.low.x);
            miny = miny.min(node.bounding_box.low.y);
            maxx = maxx.max(node.bounding_box.high.x);
            maxy = maxy.max(node.bounding_box.high.y);
            children.push(node.clone());
            size += node.size;
            sizes.push(node.size as f64);
        }

        RSTreeNode {
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
            offset: 0,
        }
    }
}

pub struct RSTree {
    root: RSTreeNode,
    data: Vec<Point>,
}

const MAX_ENTRIES_PER_LEAF: usize = 256;
const MAX_ENTRIES_PER_NODE: usize = 25;

impl RSTree {
    pub fn size(&self) -> usize {
        self.root.size() + self.data.len() * 16
    }

    pub fn from(data: &[Point]) -> RSTree {
        let mut points: Vec<Point> = Vec::new();
        points.extend_from_slice(data);
        let mut now = (points.len() as f64 / MAX_ENTRIES_PER_LEAF as f64).ceil() as usize;
        let length = points.len();
        let now_x = (now as f64).sqrt().ceil() as usize;
        let now_y = (now as f64 / now_x as f64).ceil() as usize;
        points.sort_unstable_by(|p1, p2| p1.x.partial_cmp(&p2.x).unwrap());
        let step_x = (points.len() as f64 / now_x as f64).ceil() as usize;
        let mut i = 0 as usize;
        let mut rtree_nodes: Vec<Rc<RSTreeNode>> = Vec::new();
        while i < length {
            let slice_x = &mut points[i..(i + step_x).min(length)];
            slice_x.sort_unstable_by(|p1, p2| p1.y.partial_cmp(&p2.y).unwrap());
            let step_y = (std::cmp::min(step_x, length - i) as f64 / now_y as f64).ceil() as usize;
            let mut j = 0 as usize;
            while j < slice_x.len() {
                let len = step_y.min(slice_x.len() - j);
                rtree_nodes.push(Rc::new(RSTreeNode::from_data(&slice_x[j..j+len], i + j)));
                j += len;
            }
            i += slice_x.len();
        }

        now = (rtree_nodes.len() as f64 / MAX_ENTRIES_PER_NODE as f64).ceil() as usize;
        while now > 1 {
            let now_x = (now as f64).ceil().sqrt() as usize;
            let now_y = (now as f64 / now_x as f64).ceil() as usize;
            rtree_nodes.sort_unstable_by(
                |n1, n2| (n1.bounding_box.low.x + n1.bounding_box.high.x)
                            .partial_cmp(&(n2.bounding_box.low.x + n2.bounding_box.high.x)).unwrap());
            let length = rtree_nodes.len(); 
            let step_x = (length as f64 / now_x as f64).ceil() as usize;
            let mut i = 0 as usize;
            let mut tmp_nodes: Vec<Rc<RSTreeNode>> = Vec::new();
            while i < length {
                let slice_x = &mut rtree_nodes[i..(i + step_x).min(length)];
                slice_x.sort_unstable_by(
                    |n1, n2| (n1.bounding_box.low.y + n1.bounding_box.high.y)
                                .partial_cmp(&(n2.bounding_box.low.y + n2.bounding_box.high.y)).unwrap());
                let step_y = (std::cmp::min(step_x, length - i) as f64 / now_y as f64).ceil() as usize;
                let mut j = 0 as usize;
                while j < slice_x.len() {
                    let len = step_y.min(slice_x.len() - j);
                    tmp_nodes.push(Rc::new(RSTreeNode::from_nodes(&slice_x[j..j+len])));
                    j += len;
                }
                i += slice_x.len();
            }
            rtree_nodes = tmp_nodes;
            now = (rtree_nodes.len() as f64 / MAX_ENTRIES_PER_NODE as f64) as usize;
        }

        let root = RSTreeNode::from_nodes(rtree_nodes.as_slice());
        let mut layout: Vec<Point> = Vec::new();
        let mut stack: Vec<(*mut RSTreeNode, usize)> = Vec::new();
        let root_ptr: *mut RSTreeNode = &root as *const RSTreeNode as *mut RSTreeNode;
        stack.push((root_ptr, 0));
        while !stack.is_empty() {
            let now = stack.pop().unwrap();
            let node: &mut RSTreeNode = unsafe{&mut *(now.0)};
            match &node.children {
                Some(children) => {
                    let mut offset = now.1 + node.size;
                    for child in children.iter().rev() {
                        offset -= child.size;
                        stack.push((&**child as *const RSTreeNode as *mut RSTreeNode, offset));
                    }
                }
                None => {
                    for i in node.offset..(node.offset + node.size) {
                        layout.push(points[i].clone());
                    }
                }
            }
            node.offset = now.1;
        }

        RSTree {
            root,
            data: layout,
        }
    }

    pub fn range(&self, query: &MBR) -> Vec<Point> {
        let mut res: Vec<Point> = Vec::new();
        let mut stack: Vec<&RSTreeNode> = Vec::new();
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
                    for i in now.offset..(now.offset + now.size) {
                        if query.contains(&self.data[i]) {
                            res.push(self.data[i].clone());
                        }
                    }
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
        let mut lca_root: &RSTreeNode = &self.root;
        loop {
            match &lca_root.children {
                None => { break; }
                Some(children) => {
                    let mut cnt = 0;
                    let mut tmp = lca_root;
                    for child in children.iter() {
                        if query.intersects(&child.bounding_box) {
                            cnt += 1;
                            tmp = child;
                        }
                    }
                    if cnt == 1 { lca_root = tmp; }
                    else { break; }
                }
            }
        }

        let mut rng = util::new_rng();
        let dist = Uniform::from(0.0f64..1.0f64);
        while samples.len() < k {
            let mut now: &RSTreeNode = &self.root;
            loop {
                match &now.children {
                    None => {
                        let offset = now.offset + (dist.sample(&mut rng) * (now.size as f64)) as usize;
                        if query.contains(&self.data[offset]) {
                            samples.push(self.data[offset].clone());
                        }
                        break;
                    }
                    Some(children) => {
                        let weights: Vec<f64> = children.iter().map(|node| node.size as f64).collect();
                        let alias = AliasTable::from(&weights);
                        let coin1 = dist.sample(&mut rng);
                        let coin2 = dist.sample(&mut rng);
                        now = children[alias.sample(coin1, coin2)].as_ref();
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

        // First, we found all candidate alias tables and construct a
        // spare alias table for spilled points.
        let mut candidates: Vec<&RSTreeNode> = Vec::new();
        //let mut spare: Vec<usize> = Vec::new();
        let mut stack: Vec<&RSTreeNode> = Vec::new();
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
                        candidates.push(&now);
                    }
                }
            }
        }
        //let spare_alias = AliasTable::uniform(spare.len());
        
        // Construct top level alias structure
        //let mut weights = vec![spare.len() as f64];
        let weights: Vec<f64> = candidates.iter().map(|node| node.size as f64).collect();
        let dist = Uniform::from(0.0f64..1.0f64);
        let mut rng = util::new_rng();
        let top_level_alias = AliasTable::from(&weights);

        // For each sample, construct two level sampling.
        while samples.len() < k {
            let coin1 = dist.sample(&mut rng);
            let coin2 = dist.sample(&mut rng);
            let res = top_level_alias.sample(coin1, coin2);
            let coin3 = dist.sample(&mut rng);
            let offset = (weights[res] * coin3) as usize  + candidates[res].offset;
            if query.contains(&self.data[offset]) {
                samples.push(self.data[offset].clone());
            }
        }

        samples
    }
}

