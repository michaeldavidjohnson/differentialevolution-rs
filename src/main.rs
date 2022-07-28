use rand::prelude::*;

fn main() {
    let temp = vec![0.0,1.0,-3.0,3.0];
    println!("{:?}",differential_evolution(obj,&temp,20,100,0.5,0.7));
}

fn obj(position: &Vec<f64>) -> f64{
    return (position[0] - 0.69).powf(2.0) + (position[1] - 1.7).powf(2.0);
}

fn differential_evolution(objective_function: fn(&Vec<f64>) -> f64, bounds: &Vec<f64>,population_size: i32, epochs: u32, crossover_probability: f64, differential_weight: f64) -> Vec<Vec<f64>>{
    let mut genpop = vec![];
    let mut rng = thread_rng();
    if bounds.len() % 2 != 0{
        panic!("Bounds are odd");
    }
    
    for _pop in 0..population_size {
        let mut params = vec![];
        let mut count = 0;
        for _p in 0..bounds.len()/2 {
            params.push(rng.gen_range(bounds[2*count]..bounds[2*count+1]));
            count = count + 1;
        }

        genpop.push(params);


    }

    let dist = rand::distributions::Uniform::from(1..genpop.len());
    let dist_b = rand::distributions::Uniform::from(1..genpop.len()-1);
    let dist_c = rand::distributions::Uniform::from(1..genpop.len()-2);
    let mut rng = rand::thread_rng();
    let dimensionality = genpop[0].len();

    for _ in 0..epochs{
        for idx in 0..genpop.len(){
            let mut potential = Vec::from_iter(genpop.iter().cloned());
            let mut base = Vec::from_iter(genpop.iter().cloned()); 

            let mut index = dist.sample(&mut rng);
            while index == idx{
                index = dist.sample(&mut rng);
            }
            let a = base.remove(index);

            let mut index_b = dist_b.sample(&mut rng);
            while index_b == idx || index_b == index{
                index_b = dist_b.sample(&mut rng);
            }
            let b = base.remove(index_b);

            let mut index_c = dist_c.sample(&mut rng);
            while index_c == idx || index_c == index || index_c == index_b{
                index_c = dist_c.sample(&mut rng);
            }

            let c = base.remove(index_c);

            let forced_switch = dist.sample(&mut rng);

            for val in 0..dimensionality{                
                if val == forced_switch || rng.gen_range(0.0..1.0) < crossover_probability{
                    let mut temp = a[val] + differential_weight * (b[val] - c[val]);

                    if temp > bounds[2*val+1]{
                        temp = bounds[2*val+1];
                    }
                    if temp < bounds[2*val]{
                        temp = bounds[2*val];
                    }

                    potential[idx][val] = temp;
                    
                }
            }
            let func_eval = objective_function(&potential[idx]);
            
            if func_eval < objective_function(&genpop[idx]){
                genpop[idx] = Vec::from_iter(potential[idx].iter().cloned());
            }
        }
    }

    let mut best_fun = objective_function(&genpop[0]);
    let mut best_pos = &genpop[0];
    for hmm in 0..genpop.len(){
        if objective_function(&genpop[hmm]) <= best_fun{
            best_fun = objective_function(&genpop[hmm]);
            best_pos = &genpop[hmm];
        } 
    }
    let fun = vec![best_fun];
    let bp = Vec::from_iter(best_pos.iter().cloned());
    return vec![bp, fun]
}
