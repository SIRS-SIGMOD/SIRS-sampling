use crate::geo::{MBR, Point};
use crate::alias::AliasTable;
use crate::util::WPoint;
use crate::util;
use std::rc::Rc;
use rand::distributions::{Uniform, Distribution};
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering::SeqCst;
use std::sync::Arc;
use std::thread;
use std::time::Instant;
use std::sync::mpsc::channel;
use threadpool::ThreadPool;

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
    pool: ThreadPool,
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
        let core_ids = core_affinity::get_core_ids().unwrap();
        let pool = ThreadPool::new(core_ids.len());
        for id in core_ids.into_iter() {
            pool.execute(move || {
                core_affinity::set_for_current(id);
                thread::sleep(std::time::Duration::from_secs(1));
            });
        }
        KDTree {
            root,
            data,
            pool,
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

    pub fn olken_range_sampling_naive(&self, query: &MBR, k: usize) -> Vec<Point> {
        let mut samples: Vec<Point> = Vec::new();
        if !query.intersects(&self.root.bounding_box) {
            return samples;
        }
        
        let mut rng = util::new_rng();
        let dist = Uniform::from(0.0f64..1.0f64);
        while samples.len() < k {
            let mut now: &KDTreeNode = &self.root;
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

    pub fn olken_range_sampling_throughput(&self, query: &MBR, period: u64) -> Vec<Point> {
        let flag = Arc::new(AtomicBool::new(true));
        let flag_ptr = flag.clone();
        let timer_thread = thread::spawn(move || util::timer_thread(flag_ptr, period));
        let now = Instant::now();
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
        while flag.load(SeqCst) {
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
        println!("Throughput: {}", samples.len() as f64 / now.elapsed().as_secs_f64());
        timer_thread.join().expect("Join Exception");
        samples
    }

    pub fn olken_range_sampling_naive_throughput(&self, query: &MBR, period: u64) -> Vec<Point> {
        let flag = Arc::new(AtomicBool::new(true));
        let flag_ptr = flag.clone();
        let timer_thread = thread::spawn(move || util::timer_thread(flag_ptr, period));
        let now = Instant::now();
        let mut samples: Vec<Point> = Vec::new();
        if !query.intersects(&self.root.bounding_box) {
            return samples;
        }

        let mut rng = util::new_rng();
        let dist = Uniform::from(0.0f64..1.0f64);
        while flag.load(SeqCst) {
            let mut now: &KDTreeNode = &self.root;
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
        println!("Throughput: {}", samples.len() as f64 / now.elapsed().as_secs_f64());
        timer_thread.join().expect("Join Exception");
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

    pub fn range_sampling_throughput(&self, query: &MBR, period: u64) -> Vec<Point> {
        let timer = Instant::now();
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
        let decompose_time = timer.elapsed().as_micros();

        let flag = Arc::new(AtomicBool::new(true));
        let flag_ptr = flag.clone();
        let timer_thread = thread::spawn(move || util::timer_thread(flag_ptr, period));
        // For each sample, construct two level sampling.
        let now = Instant::now();
        while flag.load(SeqCst) {
            let coin1 = dist.sample(&mut rng);
            let coin2 = dist.sample(&mut rng);
            let res = top_level_alias.sample(coin1, coin2);
            let offset = KDTree::sample_subtree(candidates[res], &mut rng, &dist);
            if query.contains(&self.data[offset].p) {
                samples.push(self.data[offset].p.clone());
            }
        }
        println!("Decomposition: {} us, Throughput: {} ops/s", decompose_time, samples.len() as f64 / now.elapsed().as_secs_f64());
        timer_thread.join().expect("Join Exception");
        samples
    }

    pub fn range_sampling_no_rej_throughput(&self, query: &MBR, period: u64) -> Vec<Point> {
        let timer = Instant::now();
        let mut samples: Vec<Point> = Vec::new();
        let mut candidates: Vec<&KDTreeNode> = Vec::new();
        let mut spare_pts: Vec<usize> = Vec::new();
        let mut spare_weights: Vec<f64> = Vec::new();
        let mut stack: Vec<&KDTreeNode> = Vec::new();
        let mut tot_spare_weight: f64 = 0.0;
        if query.intersects(&self.root.bounding_box) { stack.push(&self.root); }
        while !stack.is_empty() {
            let now = stack.pop().unwrap();
            if query.contains_mbr(&now.bounding_box) {
                candidates.push(now);
            } else {
                match &now.children {
                    None => {
                        for offset in now.start..now.end {
                            if query.contains(&self.data[offset].p) {
                                spare_pts.push(offset);
                                spare_weights.push(self.data[offset].weight);
                                tot_spare_weight += self.data[offset].weight;
                            }
                        }
                    }
                    Some((left, right)) => {
                        if query.intersects(&left.bounding_box) { stack.push(left); }
                        if query.intersects(&right.bounding_box) { stack.push(right); }
                    }
                }
            }
        }
        // Construct top level alias structure
        let mut weights: Vec<f64> = vec![tot_spare_weight as f64];
        weights.extend(candidates.iter().map(|node| node.weight));
        let spare_alias = AliasTable::from(&spare_weights);
        let dist = Uniform::from(0.0f64..1.0f64);
        let mut rng = util::new_rng();
        let top_level_alias = AliasTable::from(&weights);
        let decompose_time = timer.elapsed().as_micros();

        let flag = Arc::new(AtomicBool::new(true));
        let flag_ptr = flag.clone();
        let timer_thread = thread::spawn(move || util::timer_thread(flag_ptr, period));
        // For each sample, construct two level sampling.
        let now = Instant::now();
        while flag.load(SeqCst) {
            let coin1 = dist.sample(&mut rng);
            let coin2 = dist.sample(&mut rng);
            let res = top_level_alias.sample(coin1, coin2);
            let offset = if res == 0 {
                let coin3 = dist.sample(&mut rng);
                let coin4 = dist.sample(&mut rng);
                spare_pts[spare_alias.sample(coin3, coin4)]
            } else {
                KDTree::sample_subtree(candidates[res - 1], &mut rng, &dist)
            };
            samples.push(self.data[offset].p.clone());
        }
        println!("Decomposition: {} us, Throughput: {} ops/s", decompose_time, samples.len() as f64 / now.elapsed().as_secs_f64());
        timer_thread.join().expect("Join Exception");
        samples
    }

    fn range_sampling_olken_limited(ptr: libc::uintptr_t, query: MBR, flag: Arc<AtomicBool>) -> Vec<Point> {
        let tree = ptr as *const KDTree;
        let mut samples: Vec<Point> = Vec::new();
        let mut lca_root: &KDTreeNode = unsafe{ &(*tree).root };
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
        while flag.load(SeqCst) {
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
                        if query.contains(unsafe { &(*tree).data[offset].p }) {
                            samples.push(unsafe {&(*tree).data[offset].p}.clone());
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


    fn range_sampling_limited(ptr: libc::uintptr_t, query: MBR, candidates_ptr: libc::uintptr_t, spares_ptr: libc::uintptr_t,
        alias_ptr: libc::uintptr_t, flag: Arc<AtomicBool>) -> Vec<Point> {
        let mut samples: Vec<Point> = Vec::new();
        let tree: &KDTree = unsafe{&*(ptr as *const KDTree)};
        let dist = Uniform::from(0.0f64..1.0f64);
        let mut rng = util::new_rng();
        let top_level_alias: &AliasTable = unsafe{&*(alias_ptr as *const AliasTable)};
        let candidates: &Vec<&KDTreeNode> = unsafe{&*(candidates_ptr as *const Vec<&KDTreeNode>)};
        let spares: &Vec<&KDTreeNode> = unsafe{&*(spares_ptr as *const Vec<&KDTreeNode>)};
        while flag.load(SeqCst) {
            let coin1 = dist.sample(&mut rng);
            let coin2 = dist.sample(&mut rng);
            let res = top_level_alias.sample(coin1, coin2);
            let node: &KDTreeNode = if res < candidates.len() { candidates[res] } else { spares[res - candidates.len()] };
            let offset = KDTree::sample_subtree(node, &mut rng, &dist);
            if query.contains(&tree.data[offset].p) {
                samples.push(tree.data[offset].p.clone());
            }
        }
        samples
    }

    pub fn range_sampling_hybrid(&self, query: &MBR, period: u64) -> Vec<Point> {
        let mut samples: Vec<Point> = Vec::new();
        let flag = Arc::new(AtomicBool::new(true));
        let tree_ptr = self as *const KDTree as libc::uintptr_t;
        let query_mbr = query.clone();
        let olken_flag_pointer = flag.clone();
        let (tx1, rx1) = channel();

        self.pool.execute(move || { tx1.send(KDTree::range_sampling_olken_limited(tree_ptr, query_mbr, olken_flag_pointer)).expect("channel exception"); });
        let timer = Instant::now();
        let mut candidates: Vec<&KDTreeNode> = Vec::new();
        let mut spares: Vec<&KDTreeNode> = Vec::new();
        let mut stack: Vec<&KDTreeNode> = Vec::new();
        if query.intersects(&self.root.bounding_box) { stack.push(&self.root); }
        while !stack.is_empty() {
            let now = stack.pop().unwrap();
            if query.contains_mbr(&now.bounding_box) {
                candidates.push(now);
            } else {
                match &now.children {
                    None => {
                        spares.push(now);
                    }
                    Some((left, right)) => {
                        if query.intersects(&left.bounding_box) { stack.push(left); }
                        if query.intersects(&right.bounding_box) { stack.push(right); }
                    }
                }
            }
        }
        let mut weights: Vec<f64> = candidates.iter().map(|node| node.weight).collect();
        weights.extend(spares.iter().map(|node| node.weight));
        let top_level_alias = AliasTable::from(&weights);
        flag.store(false, SeqCst);
        let stage1_time = timer.elapsed().as_micros();
       
        flag.store(true, SeqCst);
        let query_mbr = query.clone();
        let rej_flag_pointer = flag.clone();
        let candidates_ptr = &candidates as *const Vec<&KDTreeNode> as libc::uintptr_t;
        let spares_ptr = &spares as *const Vec<&KDTreeNode> as libc::uintptr_t;
        let alias_ptr = &top_level_alias as *const AliasTable as libc::uintptr_t;
        // Level 2 fork.
        let (tx2, rx2) = channel();
        self.pool.execute(move || { tx2.send(KDTree::range_sampling_limited(tree_ptr, query_mbr, candidates_ptr, spares_ptr, alias_ptr, rej_flag_pointer)).expect("channel exception"); });
        let timer2 = Instant::now();
        let mut spare_pts: Vec<usize> = Vec::new();
        let mut spare_weights: Vec<f64> = Vec::new();
        let mut tot_spare_weight = 0f64;
        for node in spares.iter() {
            for offset in node.start..node.end {
               if query.contains(&self.data[offset].p) {
                   spare_pts.push(offset);
                   spare_weights.push(self.data[offset].weight);
                   tot_spare_weight += self.data[offset].weight;
               }
            }
        }
        let mut new_weights: Vec<f64> = vec![tot_spare_weight as f64];
        new_weights.extend(candidates.iter().map(|node| node.weight));
        let new_alias = AliasTable::from(&new_weights);
        let spare_alias = AliasTable::from(&spare_weights);
        flag.store(false, SeqCst);
        let stage2_time = timer2.elapsed().as_micros();

        let timer_flag = Arc::new(AtomicBool::new(true));
        let flag_ptr = timer_flag.clone();
        let timer_thread = thread::spawn(move || util::timer_thread(flag_ptr, period));
        let now = Instant::now();
        let dist = Uniform::from(0.0f64..1.0f64);
        let mut rng = util::new_rng();
        while timer_flag.load(SeqCst) {
            let coin1 = dist.sample(&mut rng);
            let coin2 = dist.sample(&mut rng);
            let res = new_alias.sample(coin1, coin2);
            let offset = if res == 0 {
                let coin3 = dist.sample(&mut rng);
                let coin4 = dist.sample(&mut rng);
                spare_pts[spare_alias.sample(coin3, coin4)]
            } else {
                KDTree::sample_subtree(candidates[res - 1], &mut rng, &dist)
            };
            samples.push(self.data[offset].p.clone());
        }
        let tot_time = now.elapsed().as_secs_f64();
        let tot_samples = samples.len();
        let stage1_sample = rx1.recv().expect("Join 1 exception");
        let stage2_sample = rx2.recv().expect("Join 2 exception");
        let stage1_len = stage1_sample.len();
        let stage2_len = stage2_sample.len();
        samples.extend(stage1_sample);
        samples.extend(stage2_sample);
        timer_thread.join().expect("Join exception");
        println!("Stage 1 time: {} us, Stage 1 samples: {}", stage1_time, stage1_len);
        println!("Stage 2 time: {} us, Stage 2 samples: {}", stage2_time, stage2_len);
        println!("Finish second level with throughput: {} op/s", tot_samples as f64 / tot_time);
        samples
    }
}
