use crate::alias::AliasTable;
use crate::util;
use rand::distributions::{Uniform, Distribution};

struct DyadicTreeNode {
    weight: f64,
    //interval: (usize, usize),
    left: usize,
    right: usize,
}

pub struct DyadicTree {
    nodes: Vec<DyadicTreeNode>,
    root: usize,
    len: usize,
}

impl DyadicTreeNode {
    fn from(data: &[f64], offset: usize, nodes: &mut Vec<DyadicTreeNode>) -> (usize, f64) {
        let len = data.len();
        if len == 0 {
            (0, 0.0)
        } else if len == 1 {
            let now = nodes.len();
            nodes.push(DyadicTreeNode{
                weight: data[0],
                // interval: (offset, offset + 1),
                left: 0usize,
                right: 0usize,
            });
            (now, data[0])
        } else {
            let mid = len / 2;
            let (left_node, left_weight) = DyadicTreeNode::from(&data[0..mid], offset, nodes);
            let (right_node, right_weight) = DyadicTreeNode::from(&data[mid..len], offset + mid, nodes);
            let now = nodes.len();
            nodes.push(DyadicTreeNode{
                weight: left_weight + right_weight,
                // interval: (offset, offset + len),
                left: left_node,
                right: right_node,
            });
            (now, left_weight + right_weight)
        }
    }
    
    fn from_func<T>(data: &[T], f: &dyn Fn(&T) -> f64, offset: usize, nodes: &mut Vec<DyadicTreeNode>) -> (usize, f64) {
        let len = data.len();
        if len == 0 {
            (0, 0.0)
        } else if len == 1 {
            let now = nodes.len();
            nodes.push(DyadicTreeNode{
                weight: f(&data[0]),
                // interval: (offset, offset + 1),
                left: 0usize,
                right: 0usize,
            });
            (now, f(&data[0]))
        } else {
            let mid = len / 2;
            let (left_node, left_weight) = DyadicTreeNode::from_func(&data[0..mid], f, offset, nodes);
            let (right_node, right_weight) = DyadicTreeNode::from_func(&data[mid..len], f,  offset + mid, nodes);
            let now = nodes.len();
            nodes.push(DyadicTreeNode{
                weight: left_weight + right_weight,
                // interval: (offset, offset + len),
                left: left_node,
                right: right_node,
            });
            (now, left_weight + right_weight)
        }
    }
}

impl DyadicTree {
    pub fn size(&self) -> usize {
        self.nodes.len() * std::mem::size_of::<DyadicTreeNode>()
    }

    pub fn from(data: &[f64]) -> DyadicTree {
        let mut nodes: Vec<DyadicTreeNode> = Vec::with_capacity(data.len() * 2);
        nodes.push(DyadicTreeNode {
            weight: 0.0f64,
            // interval: (0, 0),
            left: 0usize,
            right: 0usize,
        });

        let (root, _) = DyadicTreeNode::from(&data, 0, &mut nodes);
        
        DyadicTree {
            nodes,
            root,
            len: data.len(),
        }
    }
    
    pub fn from_func<T>(data: &[T], f: &dyn Fn(&T) -> f64) -> DyadicTree {
        let mut nodes: Vec<DyadicTreeNode> = Vec::with_capacity(data.len() * 2);
        nodes.push(DyadicTreeNode {
            weight: 0.0f64,
            // interval: (0, 0),
            left: 0usize,
            right: 0usize,
        });

        let (root, _) = DyadicTreeNode::from_func(&data, f, 0, &mut nodes);
        
        DyadicTree {
            nodes,
            root,
            len: data.len(),
        }
    }

    fn traverse(&self, node: usize, left: usize, right: usize, node_left: usize, node_right: usize, res: &mut Vec<(usize, usize, usize)>) {
        let now = &self.nodes[node];
        if left <= node_left && node_right <= right {
            res.push((node, node_left, node_right));
        } else {
            let mid = (node_left + node_right) / 2;
            if right <= mid {
                self.traverse(now.left, left, right, node_left, mid, res);
            } else if mid <= left {
                self.traverse(now.right, left, right, mid, node_right, res);
            } else {
                self.traverse(now.left, left, mid, node_left, mid, res);
                self.traverse(now.right, mid, right, mid, node_right, res);
            }
        }
    }

    pub fn single_sample(&self, left: usize, right: usize) -> usize {
        let mut candidates: Vec<(usize, usize, usize)> = Vec::new();
        self.traverse(self.root, left, right, 0, self.len, &mut candidates);
        let weights: Vec<f64> = candidates.iter().map(|x| self.nodes[x.0].weight).collect();
        let alias = AliasTable::from(&weights);
        let mut rng = util::new_rng();
        let dist = Uniform::from(0.0f64..1.0f64);
        let coin1 = dist.sample(&mut rng);
        let coin2 = dist.sample(&mut rng);
        let root_candidate = candidates[alias.sample(coin1, coin2)];
        let mut node: &DyadicTreeNode = &self.nodes[root_candidate.0];
        let mut node_left = root_candidate.1;
        let mut node_right = root_candidate.2;
        while node_right - node_left > 1 {
            let left_weight = self.nodes[node.left].weight;
            let thres = left_weight / node.weight;
            let mid = (node_left + node_right) / 2;
            if dist.sample(&mut rng) <= thres {
                node = &self.nodes[node.left];
                node_right = mid;
            } else {
                node = &self.nodes[node.right];
                node_left = mid;
            }
        }
        node_left
    }
    
    pub fn sample(&self, left: usize, right: usize, k: usize) -> Vec<usize> {
        let mut candidates: Vec<(usize, usize, usize)> = Vec::new();
        self.traverse(self.root, left, right, 0, self.len, &mut candidates);
        let weights: Vec<f64> = candidates.iter().map(|x| self.nodes[x.0].weight).collect();
        let alias = AliasTable::from(&weights);
        let mut rng = util::new_rng();
        let dist = Uniform::from(0.0f64..1.0f64);
        let mut samples: Vec<usize> = Vec::new();
        for _ in 0..k {
            let coin1 = dist.sample(&mut rng);
            let coin2 = dist.sample(&mut rng);
            let root_candidate = candidates[alias.sample(coin1, coin2)];
            let mut node: &DyadicTreeNode = &self.nodes[root_candidate.0];
            let mut node_left = root_candidate.1;
            let mut node_right = root_candidate.2;
            while node_right - node_left > 1 {
                let left_weight = self.nodes[node.left].weight;
                let thres = left_weight / node.weight;
                let mid = (node_left + node_right) / 2;
                if dist.sample(&mut rng) <= thres {
                    node = &self.nodes[node.left];
                    node_right = mid;
                } else {
                    node = &self.nodes[node.right];
                    node_left = mid;
                }
            }
            samples.push(node_left)
        }
        samples
    }
}
