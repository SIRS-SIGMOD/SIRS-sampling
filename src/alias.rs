use std::fmt;

pub struct AliasTable {
    n: usize,
    cutoff: Vec<f64>,
    alias: Vec<usize>,
}

impl fmt::Display for AliasTable {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "(cutoff = {:?}, \n alias = {:?})", self.cutoff, self.alias)
    }
}

impl AliasTable {
    pub fn uniform(n: usize) -> AliasTable {
        AliasTable {
            n,
            cutoff: vec![1.0; n],
            alias: (0..n).collect(),

        }
    }

    pub fn from(weight: &[f64]) -> AliasTable {
        let sum_weight: f64 = weight.iter().sum();
        let mut norm_weight: Vec<f64> =
            weight.iter().map(|x| x * weight.len() as f64 / sum_weight).collect();
        
        let mut overfull: Vec<usize> = Vec::new();
        let mut underfull: Vec<usize> = Vec::new();
        let mut cutoff: Vec<f64> = vec![1.0; weight.len()];
        let mut alias: Vec<usize> = (0..weight.len()).collect();
        let mut k: usize = weight.len() - 1;
        loop {
            if norm_weight[k] > 1.0  {
                overfull.push(k);
            } else if norm_weight[k] < 1.0 {
                underfull.push(k);
            }
            if k == 0 {
                break;
            }
            k -= 1;
        }

        while !overfull.is_empty() && !underfull.is_empty() {
            let current_overfull = overfull.pop().unwrap();
            let current_underfull = underfull.pop().unwrap();
            cutoff[current_underfull] = norm_weight[current_underfull];
            alias[current_underfull] = current_overfull;
            norm_weight[current_overfull] = norm_weight[current_overfull] + norm_weight[current_underfull] - 1.0;
            if norm_weight[current_overfull] > 1.0 {
                overfull.push(current_overfull);
            } else if norm_weight[current_overfull] < 1.0 {
                underfull.push(current_overfull);
            }
        }

        AliasTable {
            n: weight.len(),
            cutoff: cutoff,
            alias: alias,
        }
    }

    pub fn sample(&self, coin1: f64, coin2: f64) -> usize {
        let k: usize = (self.n as f64 * coin1) as usize;
        if coin2 < self.cutoff[k] {
            k
        } else {
            self.alias[k]
        }
    }
}