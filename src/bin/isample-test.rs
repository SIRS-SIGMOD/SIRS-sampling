use range_sampling::isample::DyadicTree;
use std::collections::HashMap;

fn main() {
    let my_array : Vec<f64> = (0..10000).map(|ix| {
        // 1 (inclusive) to 21 (exclusive)
        ix as f64
    }).collect();

    let array_sample = DyadicTree::from(&my_array);
    let samples = array_sample.sample(3, 20, 1870000);

    let mut map: HashMap<usize, usize> = HashMap::new();
    for sample in samples.iter() {
        let entry = map.entry(*sample).or_insert(0);
        *entry += 1;
    }
    println!("{:?}", map );
}