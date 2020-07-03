
use crate::geo::{MBR, Point};
use crate::alias::AliasTable;
use crate::util;
use std::rc::Rc;
use rand::distributions::{Uniform, Distribution};

const MAX_ENTRIES_PER_LEAF: usize = 256;
const MAX_ENTRIES_PER_NODE: usize = 25;
const RSB_SAMPLE_BUFFER_SIZE: usize = 128;

struct RSBTreeNode {
    bounding_box: MBR,
    size: usize,
    children: Option<Vec<Rc<RSBTreeNode>>>,
    offset: usize,
    sample_buffer: Vec<Point>,
    valid_ptr: usize,
}

impl RSBTreeNode {
    fn from_data(data: &[Point], offset: usize) -> RSBTreeNode {
        let mut minx = std::f64::MAX;
        let mut miny = std::f64::MAX;
        let mut minz = std::f64::MAX;
        let mut maxx = std::f64::MIN;
        let mut maxy = std::f64::MIN;
        let mut maxz = std::f64::MIN;
        for p in data.iter() {
            minx = minx.min(p.x); miny = miny.min(p.y); minz = minz.min(p.z);
            maxx = maxx.max(p.x); maxy = maxy.max(p.y); maxz = maxy.max(p.z);
        }

        RSBTreeNode {
            bounding_box: MBR {
                low: Point {
                    x: minx,
                    y: miny,
                    z: minz,
                },
                high: Point {
                    x: maxx,
                    y: maxy,
                    z: maxz,
                }
            },
            size: data.len(),
            children: None,
            offset,
            sample_buffer: Vec::new(),
            valid_ptr: 0,
        }
    }

    fn from_nodes(nodes: &[Rc<RSBTreeNode>]) -> RSBTreeNode {
        let mut minx = std::f64::MAX;
        let mut miny = std::f64::MAX;
        let mut minz = std::f64::MAX;
        let mut maxx = std::f64::MIN;
        let mut maxy = std::f64::MIN;
        let mut maxz = std::f64::MIN;
        let mut size = 0_usize;
        let mut children: Vec<Rc<RSBTreeNode>> = Vec::new();
        let mut sizes: Vec<f64> = Vec::new();
        for node in nodes.iter() {
            minx = minx.min(node.bounding_box.low.x);
            miny = miny.min(node.bounding_box.low.y);
            minz = minz.min(node.bounding_box.low.z);
            maxx = maxx.max(node.bounding_box.high.x);
            maxy = maxy.max(node.bounding_box.high.y);
            maxz = maxz.max(node.bounding_box.high.z);
            children.push(node.clone());
            size += node.size;
            sizes.push(node.size as f64);
        }

        RSBTreeNode {
            bounding_box: MBR {
                low: Point {
                    x: minx,
                    y: miny,
                    z: minz,
                },
                high: Point {
                    x: maxx,
                    y: maxy,
                    z: maxz,
                }
            },
            size,
            children: Some(children),
            offset: 0,
            sample_buffer: Vec::new(),
            valid_ptr: 0,
        }
    }
}

pub struct RSBTree {
    root: RSBTreeNode,
    data: Vec<Point>,
}


impl RSBTree {
    pub fn from(data: &[Point]) -> RSBTree {
        let mut points: Vec<Point> = Vec::new();
        points.extend_from_slice(data);
        let mut now = (points.len() as f64 / MAX_ENTRIES_PER_LEAF as f64).ceil() as usize;
        let length = points.len();
        let now_x = (now as f64).cbrt().ceil() as usize;
        let now_y = (now as f64).cbrt().ceil() as usize;
        let now_z = (now as f64 / now_x as f64 / now_y as f64).ceil() as usize;
        points.sort_unstable_by(|p1, p2| p1.x.partial_cmp(&p2.x).unwrap());
        let step_x = (points.len() as f64 / now_x as f64).ceil() as usize;
        let mut i = 0 as usize;
        let mut rtree_nodes: Vec<Rc<RSBTreeNode>> = Vec::new();
        while i < length {
            let slice_x = &mut points[i..(i + step_x).min(length)];
            slice_x.sort_unstable_by(|p1, p2| p1.y.partial_cmp(&p2.y).unwrap());
            let step_y = (std::cmp::min(step_x, length - i) as f64 / now_y as f64).ceil() as usize;
            let mut j = 0 as usize;
            let length_x = slice_x.len();
            while j < length_x {
                let slice_y = &mut slice_x[j..(j + step_y).min(length_x)];
                slice_y.sort_unstable_by(|p1, p2| p1.z.partial_cmp(&p2.z).unwrap());
                let step_z = (std::cmp::min(step_y, length_x - j) as f64 / now_z as f64).ceil() as usize;
                let mut k = 0 as usize;
                while k < slice_y.len() {
                    let len = step_z.min(slice_y.len() - k);
                    rtree_nodes.push(Rc::new(RSBTreeNode::from_data(&slice_y[k..k+len], i + j + k)));
                    k += len;
                }
                j += slice_y.len();
            }
            i += length_x;
        }

        now = (rtree_nodes.len() as f64 / MAX_ENTRIES_PER_NODE as f64).ceil() as usize;
        while now > 1 {
            let now_x = (now as f64).ceil().cbrt() as usize;
            let now_y = (now as f64).ceil().cbrt() as usize;
            let now_z = (now as f64 / now_x as f64 / now_y as f64).ceil() as usize;
            rtree_nodes.sort_unstable_by(
                |n1, n2| (n1.bounding_box.low.x + n1.bounding_box.high.x)
                            .partial_cmp(&(n2.bounding_box.low.x + n2.bounding_box.high.x)).unwrap());
            let length = rtree_nodes.len(); 
            let step_x = (length as f64 / now_x as f64).ceil() as usize;
            let mut i = 0 as usize;
            let mut tmp_nodes: Vec<Rc<RSBTreeNode>> = Vec::new();
            while i < length {
                let slice_x = &mut rtree_nodes[i..(i + step_x).min(length)];
                slice_x.sort_unstable_by(
                    |n1, n2| (n1.bounding_box.low.y + n1.bounding_box.high.y)
                                .partial_cmp(&(n2.bounding_box.low.y + n2.bounding_box.high.y)).unwrap());
                let step_y = (std::cmp::min(step_x, length - i) as f64 / now_y as f64).ceil() as usize;
                let mut j = 0 as usize;
                let length_x = slice_x.len();
                while j < length_x {
                    let slice_y = &mut slice_x[j..(j + step_y).min(length_x)];
                    slice_y.sort_unstable_by(
                        |n1, n2| (n1.bounding_box.low.z + n1.bounding_box.high.z)
                                .partial_cmp(&(n2.bounding_box.low.z + n2.bounding_box.high.z)).unwrap());
                    let step_z = (std::cmp::min(step_y, length_x - j) as f64 / now_z as f64).ceil() as usize;
                    let mut k = 0 as usize;
                    while k < slice_y.len() {
                        let len = step_z.min(slice_y.len() - k);
                        tmp_nodes.push(Rc::new(RSBTreeNode::from_nodes(&slice_y[k..k+len])));
                        k += len;
                    }
                    j += slice_y.len();
                }
                i += length_x;
            }
            rtree_nodes = tmp_nodes;
            now = (rtree_nodes.len() as f64 / MAX_ENTRIES_PER_NODE as f64) as usize;
        }

        let mut root = RSBTreeNode::from_nodes(rtree_nodes.as_slice());
        let mut layout: Vec<Point> = Vec::new();
        let mut stack: Vec<(*mut RSBTreeNode, usize)> = Vec::new();
        let root_ptr: *mut RSBTreeNode = &mut root as *mut RSBTreeNode;
        stack.push((root_ptr, 0));
        while !stack.is_empty() {
            let now = stack.pop().unwrap();
            let node: &mut RSBTreeNode = unsafe{&mut *(now.0)};
            match &node.children {
                Some(children) => {
                    let mut offset = now.1 + node.size;
                    for child in children.iter().rev() {
                        offset -= child.size;
                        stack.push((&**child as *const RSBTreeNode as *mut RSBTreeNode, offset));
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
        assert!(stack.is_empty());
        stack.push((root_ptr, 0));
        while !stack.is_empty() {
            let now = stack.pop().unwrap();
            let node: &mut RSBTreeNode = unsafe{&mut *(now.0)};
            if let Some(children) = &node.children {
                let mut offset = now.1 + node.size;
                for child in children.iter().rev() {
                    offset -= child.size;
                    stack.push((&**child as *const RSBTreeNode as *mut RSBTreeNode, offset));
                }
                node.sample_buffer = util::sample_from(&layout[node.offset..(node.offset + node.size)], RSB_SAMPLE_BUFFER_SIZE);
            }
        }

        RSBTree {
            root,
            data: layout,
        }
    }

    pub fn range(&self, query: &MBR) -> Vec<Point> {
        let mut res: Vec<Point> = Vec::new();
        let mut stack: Vec<&RSBTreeNode> = Vec::new();
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

    pub fn range_sampling(&mut self, query: &MBR, k: usize) -> Vec<Point> {
        let mut samples: Vec<Point> = Vec::new();
        let mut frontier: Vec<*mut RSBTreeNode> = Vec::new();
        let mut new_frontier: Vec<*mut RSBTreeNode> = Vec::new();
        let mut weights: Vec<f64> = Vec::new();
        frontier.push(&mut self.root as *mut RSBTreeNode);
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
                    let sample = &self.data[node.offset + (node.size as f64 * coin) as usize];
                    if query.contains(sample) { samples.push(sample.clone()); }
                }
                Some(children) => {
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
                                for child in children.iter() {
                                    let child_ptr = (&**child as *const RSBTreeNode) as *mut RSBTreeNode;
                                    if child.bounding_box.intersects(&query) {
                                        new_frontier.push(child_ptr);
                                        weights.push(child.size  as f64);
                                    }
                                }
                            } else {
                                new_frontier.push(item.clone());
                                weights.push(unsafe {(**item).size} as f64);
                            }
                        }
                        //replenish buffer
                        node.sample_buffer = util::sample_from(&self.data[node.offset..(node.offset + node.size)], RSB_SAMPLE_BUFFER_SIZE);
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
