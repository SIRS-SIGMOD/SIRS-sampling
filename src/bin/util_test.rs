extern crate range_sampling;

use range_sampling::geo::{MBR, Point};
use range_sampling::alias::AliasTable;
use rand::distributions::{Uniform, Distribution};

fn main() {
    let low = Point::new(10.0, 20.0, 40.0);
    let high = Point::new(20.0, 30.0, 50.0);
    let mbr = MBR::new(&low, &high);
    println!("{}", mbr);
    println!("diagnal length: {}", low.min_dist(&high));
    let p1 = Point::new(15.0, 25.0, 45.0);
    let p2 = Point::new(0.0, 0.0, 0.0);
    println!("{} is inside {}?: {}", p1, mbr, mbr.contains(&p1));
    println!("{} is inside {}?: {}", p2, mbr, mbr.contains(&p2));

    let test_point = Point::new(-82.080596, 39.40811, 4000.0);
    let zv = test_point.to_zvalue();
    let tmp_p = Point::from_zvalue(zv);
    println!("{} {}", tmp_p, test_point);
    assert_eq!(tmp_p, test_point);

    let weights: Vec<f64> = vec![1.0, 1.0, 5.0, 3.0];
    let alias_table = AliasTable::from(&weights);
    println!("{}", alias_table);

    let mut stats: Vec<usize> = vec![0; 4];
    let dist = Uniform::from(0.0f64..1.0f64);
    let mut rng = rand::thread_rng();
    for _ in 0..1000000 {
        let coin1 = dist.sample(&mut rng);
        let coin2 = dist.sample(&mut rng);
        stats[alias_table.sample(coin1, coin2)] += 1;
    }
    println!("{:?}", stats);

}