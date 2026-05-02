use std::time::Instant;

fn lcg_next(value: u64) -> u64 {
    const A: u64 = 1664525;
    const C: u64 = 1013904223;
    const M: u64 = 1u64 << 32;
    (A.wrapping_mul(value).wrapping_add(C)) % M
}

fn max_subarray_sum(n: usize, seed: u64, min_val: i64, max_val: i64) -> i64 {
    let range = (max_val - min_val + 1) as u64;
    
    // Generate random numbers
    let mut random_numbers = Vec::with_capacity(n);
    let mut value = seed;
    for _ in 0..n {
        value = lcg_next(value);
        random_numbers.push((value % range) as i64 + min_val);
    }
    
    // Kadane's algorithm for maximum subarray sum
    let mut max_sum = i64::MIN;
    let mut current_sum = 0i64;
    
    for &num in &random_numbers {
        current_sum = current_sum.saturating_add(num).max(num);
        max_sum = max_sum.max(current_sum);
    }
    
    max_sum
}

fn total_max_subarray_sum(n: usize, initial_seed: u64, min_val: i64, max_val: i64) -> i64 {
    let mut total_sum = 0i64;
    let mut seed = initial_seed;
    
    for _ in 0..20 {
        seed = lcg_next(seed);
        total_sum += max_subarray_sum(n, seed, min_val, max_val);
    }
    
    total_sum
}

fn main() {
    let n = 10000;
    let initial_seed = 42u64;
    let min_val = -10i64;
    let max_val = 10i64;
    
    let start_time = Instant::now();
    let result = total_max_subarray_sum(n, initial_seed, min_val, max_val);
    let end_time = Instant::now();
    
    println!("Total Maximum Subarray Sum (20 runs): {}", result);
    println!("Execution Time: {:.6} seconds", (end_time - start_time).as_secs_f64());
}