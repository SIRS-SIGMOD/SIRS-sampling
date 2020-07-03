use crate::geo::{MBR, Point};
use crate::alias::AliasTable;
use crate::util;
use std::rc::Rc;
use rand::distributions::{Uniform, Distribution};

enum RSTreeEntries {
    Leaves(Vec<Point>),
    Nodes(Vec<Rc<RSTreeNode>>),
}

struct RSTreeNode {
    bounding_box: MBR,
    size: usize,
    children: RSTreeEntries,
    alias: AliasTable,
}

impl RSTreeNode {
    fn from_data(data: &[Point]) -> RSTreeNode {
        let mut minx = std::f64::MAX;
        let mut miny = std::f64::MAX;
        let mut maxx = std::f64::MIN;
        let mut maxy = std::f64::MIN;
        let mut children: Vec<Point> = Vec::new();
        for p in data.iter() {
            minx = minx.min(p.x);
            miny = miny.min(p.y);
            maxx = maxx.max(p.x);
            maxy = maxy.max(p.y);
            children.push(p.clone());
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
            children: RSTreeEntries::Leaves(children),
            alias: AliasTable::uniform(data.len()),
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
            children: RSTreeEntries::Nodes(children),
            alias: AliasTable::from(&sizes),
        }
    }
}

pub struct RSTree {
    root: RSTreeNode,
}

const MAX_ENTRIES_PER_NODE: usize = 25;

impl RSTree {
    pub fn from(points: &mut [Point]) -> RSTree {
        let mut now = (points.len() as f64 / MAX_ENTRIES_PER_NODE as f64).ceil() as usize;
        let length = points.len();
        let now_x = (now as f64).sqrt().ceil() as usize;
        let now_y = (now as f64 / now_x as f64).ceil() as usize;
        points.sort_unstable_by(|p1, p2| p1.x.partial_cmp(&p2.x).unwrap());
        let step_x = (points.len() as f64 / now_x as f64).ceil() as usize;
        let mut i = 0 as usize;
        let mut rtree_nodes: Vec<Rc<RSTreeNode>> = Vec::new();
        while i < length {
            let data = &mut points[i..(i + step_x).min(length)];
            data.sort_unstable_by(|p1, p2| p1.y.partial_cmp(&p2.y).unwrap());
            let step_y = (std::cmp::min(step_x, length - i) as f64 / now_y as f64).ceil() as usize;
            let mut j = 0 as usize;
            while j < data.len() {
                let len = step_y.min(data.len() - j);
                rtree_nodes.push(Rc::new(RSTreeNode::from_data(&data[j..j+len])));
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
            let mut tmp_nodes: Vec<Rc<RSTreeNode>> = Vec::new();
            while i < length {
                let data = &mut rtree_nodes[i..(i + step_x).min(length)];
                data.sort_unstable_by(
                    |n1, n2| (n1.bounding_box.low.y + n1.bounding_box.high.y)
                                .partial_cmp(&(n2.bounding_box.low.y + n2.bounding_box.high.y)).unwrap());
                let step_y = (std::cmp::min(step_x, length - i) as f64 / now_y as f64).ceil() as usize;
                let mut j = 0 as usize;
                while j < data.len() {
                    let len = step_y.min(data.len() - j);
                    tmp_nodes.push(Rc::new(RSTreeNode::from_nodes(&data[j..j+len])));
                    j += len;
                }
                i += data.len();
            }
            rtree_nodes = tmp_nodes;
            now = (rtree_nodes.len() as f64 / MAX_ENTRIES_PER_NODE as f64) as usize;
        }

        RSTree {
            root: RSTreeNode::from_nodes(rtree_nodes.as_slice()),
        }
    }

    pub fn range_sampling(&self, query:&MBR, k: usize) -> Vec<Point> {
        let mut samples: Vec<Point> = Vec::new();
        if !query.intersects(&self.root.bounding_box) {
            return samples;
        }
        let mut lca_root: &RSTreeNode = &self.root;
        loop {
            match &lca_root.children {
                RSTreeEntries::Leaves(_) => { break; }
                RSTreeEntries::Nodes(nodes) => {
                    let mut cnt = 0;
                    let mut tmp = lca_root;
                    for node in nodes.iter() {
                        if node.bounding_box.intersects(&query) {
                            cnt += 1;
                            tmp = &node;
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
            let mut now: &RSTreeNode = &lca_root; 
            loop {
                match &now.children {
                    RSTreeEntries::Leaves(leaves) => {
                        let offset = (dist.sample(&mut rng) * leaves.len() as f64) as usize;
                        if query.contains(&leaves[offset]) {
                            samples.push(leaves[offset].clone());
                        }
                        break;
                    }
                    RSTreeEntries::Nodes(nodes) => {
                        let coin1 = dist.sample(&mut rng);
                        let coin2 = dist.sample(&mut rng);
                        let next_node = &nodes[now.alias.sample(coin1, coin2)];
                        if !query.intersects(&next_node.bounding_box) {
                            //Reject if bounding box does not overlap.
                            break;
                        } else {
                            now = next_node;
                        }
                    }
                }
            }
        }
        samples
    }
}
