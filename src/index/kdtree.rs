use crate::geo::{MBR, Point};
use crate::alias::AliasTable;
use crate::util;
use rand::distributions::{Uniform, Distribution};
use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::sync::atomic::Ordering::SeqCst;
use std::sync::Arc;
use std::rc::Rc;
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
    start: usize,
    end: usize,
}

pub struct KDTree {
    root: KDTreeNode,
    data: Vec<Point>,
    pool: ThreadPool,
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
            if level % 2 == 0 {
                let split = order_stat::kth_by(points, mid, |p1, p2| p1.x.partial_cmp(&p2.x).unwrap());
                left_bounding_box.high.x = split.x;
                right_bounding_box.low.x = split.x;
            } else {
                let split = order_stat::kth_by(points, mid, |p1, p2| p1.y.partial_cmp(&p2.y).unwrap());
                left_bounding_box.high.y = split.y;
                right_bounding_box.low.y = split.y;
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
        48 + if let Some((left, right)) = &self.children {
            16 + left.size() + right.size()
        } else { 0 }
    }
}

impl KDTree {
    pub fn from(data: &[Point]) -> KDTree {
        let mut points_data: Vec<Point> = Vec::new();
        points_data.extend_from_slice(&data);
        let root = KDTreeNode::new(&mut points_data, 0, 0, data.len(), MBR::from_points(data));
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
            data: points_data,
            pool
        }
    }

    pub fn size(&self) -> usize {
        &self.data.len() * 16 + self.root.size()
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
            let mut now: &KDTreeNode = lca_root;
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


    pub fn range_sampling_olken_limited(ptr: libc::uintptr_t, query: MBR, flag: Arc<AtomicBool>) -> Vec<Point> {
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
                        let offset = (dist.sample(&mut rng) * (now.end - now.start) as f64) as usize + now.start;
                        if query.contains(unsafe { &(*tree).data[offset] }) {
                            samples.push(unsafe{(*tree).data[offset].clone()});
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
            let coin3 = dist.sample(&mut rng);
            let node: &KDTreeNode = if res < candidates.len() { candidates[res] } else { spares[res - candidates.len()] };
            let offset = ((node.end - node.start) as f64 * coin3) as usize  + node.start;
            if query.contains(&tree.data[offset]) {
                samples.push(tree.data[offset].clone());
            }
        }
        samples
    }
    
    pub fn range_sampling_hybrid(&self, query: &MBR, period: u64) -> Vec<Point> {
        let mut samples: Vec<Point> = Vec::new();
        let tot_samples = Arc::new(AtomicUsize::new(0));
        let flag = Arc::new(AtomicBool::new(true));
        let tree_ptr = self as *const KDTree as libc::uintptr_t;
        let query_mbr = query.clone();
        let oklen_flag_pointer = flag.clone();
        let (tx1, rx1) = channel();
        // Level 1 fork:
        self.pool.execute(move || { tx1.send(KDTree::range_sampling_olken_limited(tree_ptr, query_mbr, oklen_flag_pointer)).expect("channel exception"); });
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
        let mut weights: Vec<f64> = candidates.iter().map(|node| (node.end - node.start) as f64).collect();
        weights.extend(spares.iter().map(|node| (node.end - node.start) as f64));
        let top_level_alias = AliasTable::from(&weights);
        flag.store(false, SeqCst);
        // let stage1_samples = tot_samples.load(SeqCst);
        let stage1_time = timer.elapsed().as_micros();

        tot_samples.store(0, SeqCst);
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
        for node in spares.iter() {
            for offset in node.start..node.end {
               if query.contains(&self.data[offset]) {
                   spare_pts.push(offset);
               }
            }
        }
        let mut new_weights: Vec<f64> = vec![spare_pts.len() as f64];
        new_weights.extend(candidates.iter().map(|node| (node.end - node.start) as f64));
        let new_alias = AliasTable::from(&new_weights);
        flag.store(false, SeqCst);
        let stage2_time = timer2.elapsed().as_micros();
        // let stage2_samples = tot_samples.load(SeqCst);
        
        // Level 3 Fork.
        tot_samples.store(0, SeqCst);
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
            let coin3 = dist.sample(&mut rng);
            let offset = if res == 0 {
                spare_pts[(coin3 * spare_pts.len() as f64) as usize]
            } else {
                (new_weights[res] * coin3) as usize + candidates[res - 1].start
            };
            samples.push(self.data[offset].clone());
            tot_samples.fetch_add(1, SeqCst);
        }
        let tot_time = now.elapsed().as_secs_f64();
        let stage1_sample = rx1.recv().expect("Join 1 exception");
        let stage2_sample = rx2.recv().expect("Join 2 exception");
        let stage1_len = stage1_sample.len();
        let stage2_len = stage2_sample.len();
        samples.extend(stage1_sample);
        samples.extend(stage2_sample);
        timer_thread.join().expect("Join exception");
        println!("Stage 1 time: {} us, Stage 1 samples: {}", stage1_time, stage1_len);
        println!("Stage 2 time: {} us, Stage 2 samples: {}", stage2_time, stage2_len);
        println!("Finish second level with throughput: {} op/s", tot_samples.load(SeqCst) as f64 / tot_time);
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
        println!("Throughput: {}", samples.len() as f64 / now.elapsed().as_secs_f64());
        timer_thread.join().expect("Join Exception");
        samples
    }

    pub fn range_sampling_throughput(&self, query: &MBR, period: u64) -> Vec<Point> {
        let timer = Instant::now();
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
            let coin3 = dist.sample(&mut rng);
            let offset = (weights[res] * coin3) as usize  + candidates[res].start;
            if query.contains(&self.data[offset]) {
                samples.push(self.data[offset].clone());
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
        let mut spare: Vec<usize> = Vec::new();
        let mut stack: Vec<&KDTreeNode> = Vec::new();
        if query.intersects(&self.root.bounding_box) { stack.push(&self.root); }
        while !stack.is_empty() {
            let now = stack.pop().unwrap();
            if query.contains_mbr(&now.bounding_box) {
                candidates.push(now);
            } else {
                match &now.children {
                    None => {
                        for i in now.start..now.end {
                            if query.contains(&self.data[i]) {
                                spare.push(i);
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
        let mut weights = vec![spare.len() as f64];
        weights.extend(candidates.iter().map(|node| (node.end - node.start) as f64));
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
            let coin3 = dist.sample(&mut rng);
            let offset = if res == 0 {
                spare[(coin3 * spare.len() as f64) as usize]
            } else { (weights[res] * coin3) as usize  + candidates[res - 1].start };
            samples.push(self.data[offset].clone());
        }

        println!("Decomposition: {} us, Throughput: {} ops/s", decompose_time, samples.len() as f64 / now.elapsed().as_secs_f64());
        timer_thread.join().expect("Join Exception");
        samples
    }
}