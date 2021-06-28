/**
 * Author:  Daniel Garcia (RetroDISTORT)
 * Date:    June 27, 2021
 * Licence: GPL-3.0 License
 *
 * Description:
 *  Implementation of a variable sized Deep Feed Forward (DFF) Neural Network.
 *
 * Credits: 
 *  About Medicine: https://www.youtube.com/channel/UCa_k1nTwPlKIn5YUytp4Nzw
 *  Justin Johnson: https://github.com/jcjohnson
 *
 * Notes:
 *  Create a new sample project:
 *  cargo new test_project && cd test_project && cargo build && cargo run
 *
 *  Run the project:println!("{:?}",gradient_h_values);
 *   cargo run
 */

extern crate rand; //Used for rand
use rand::distributions::{Normal};

const INPUT_NODES: usize = 2;
const OUTPUT_NODES: usize = 2;
const HIDDEN_NODES: usize = 3;
const BATCH_SIZE: usize = 8;

fn main(){
    //let args: Vec<String> = env::args().collect(); //[][inputs][hidden layers][neurons per layer][output]
    let mut inputData = vec![vec![0.0; INPUT_NODES]; BATCH_SIZE];
    let mut outputData = vec![vec![0.0; OUTPUT_NODES]; BATCH_SIZE];
    let mut weightMatrix_1 = vec![vec![0.0; HIDDEN_NODES]; INPUT_NODES];
    let mut weightMatrix_2 = vec![vec![0.0; OUTPUT_NODES]; HIDDEN_NODES];
    
    get_data(&mut weightMatrix_1);
    get_data(&mut weightMatrix_2);
    get_data(&mut inputData);
    get_data(&mut weightMatrix_1);

    for index in 0..1000
    { 
	let mut h_values = dot_product(&mut inputData, &mut weightMatrix_1);
	let mut h_relu = rectified_linear_unit(&mut h_values);
	let mut output_data_predictions = dot_product(&mut h_relu, &mut weightMatrix_2);
	let mut loss = sum(&mut square(&mut subtract(&mut output_data_predictions, &mut outputData)));
	let mut gradient_predictions = multiply(&mut subtract(&mut output_data_predictions, &mut outputData), 2.0);
	let mut gradient_w2 = dot_product(&mut transpose(&mut h_relu), &mut gradient_predictions);
	let mut gradient_h_relu = dot_product(&mut gradient_predictions, &mut transpose(&mut weightMatrix_2));
	let mut gradient_h_values = replace(&mut gradient_h_relu, &mut h_values); //gradient_h_values[h_values<0]=0
	let mut gradient_w1 = dot_product(&mut transpose(&mut inputData), &mut gradient_h_values);
	weightMatrix_1 = subtract(&mut weightMatrix_1, &mut multiply(&mut gradient_w1, 0.001));
	weightMatrix_2 = subtract(&mut weightMatrix_2, &mut multiply(&mut gradient_w2, 0.001));

	println!("{:?}", loss);
    }
}

fn dot_product(v1: &mut Vec<Vec<f64>>, v2: &mut Vec<Vec<f64>>) -> Vec<Vec<f64>>
{
    assert!(v1[0].len() == v2.len()); //check if dot product can be done with the given matricies
    
    let mut result = vec![vec![0.0; v2[1].len()]; v1.len()];
    
    for index in 0..v1.len(){
	for subIndex in 0..v2[0].len(){
	    let mut sum = 0.0;
	    for matrixIndex in 0..v1[1].len(){
		sum += v1[index][matrixIndex] * v2[matrixIndex][subIndex];
	    }
	    result[index][subIndex] = sum as f64;
	}
    }
    return(result);
}

fn replace(v1: &mut Vec<Vec<f64>>, v2: &mut Vec<Vec<f64>>) -> Vec<Vec<f64>>
{
    assert!(v1.len() == v2.len() && v1[0].len() == v2[0].len()); //check if dot product can be done with the given matricies
    
    let mut result = vec![vec![0.0; v1[0].len()]; v1.len()];
    
    for x in 0..v1.len(){
	for y in 0..v1[0].len(){
	    result[x][y] = if v2[x][y] < 0.0 { 0.0 } else { v1[x][y] };
	}
    }
    return(result);
}

fn subtract(v1: &mut Vec<Vec<f64>>, v2: &mut Vec<Vec<f64>>) -> Vec<Vec<f64>>
{
    assert!(v1.len() == v2.len() && v1[0].len() == v2[0].len()); //check if dot product can be done with the given matricies
    
    let mut result = vec![vec![0.0; v1[0].len()]; v1.len()];
    
    for index in 0..v1.len(){
	for subIndex in 0..v1[0].len(){
	    result[index][subIndex] = v1[index][subIndex] - v2[index][subIndex];
	}
    }
    return(result);
}

fn transpose(v1: &mut Vec<Vec<f64>>) -> Vec<Vec<f64>>
{
    let mut result = vec![vec![0.0; v1.len()]; v1[0].len()];
    
    for x in 0..result.len(){
	for y in 0..result[0].len(){
	    result[x][y] = v1[y][x];
	}
    }
    return(result);
}

fn square(v1: &mut Vec<Vec<f64>>) -> Vec<Vec<f64>>
{
    let mut result = vec![vec![0.0; v1[0].len()]; v1.len()];
    
    for x in 0..result.len(){
	for y in 0..result[0].len(){
	    result[x][y] = v1[x][y]*v1[x][y];
	}
    }
    return(result);
}

fn sum(v1: &mut Vec<Vec<f64>>) -> f64
{
    let mut result: f64 = 0.0;
    
    for x in 0..v1.len(){
	for y in 0..v1[0].len(){
	    result += v1[x][y];
	}
    }
    return(result);
}

fn multiply(v1: &mut Vec<Vec<f64>>, val: f64) -> Vec<Vec<f64>>
{
    let mut result = vec![vec![0.0; v1[0].len()]; v1.len()];
    
    for x in 0..result.len(){
	for y in 0..result[0].len(){
	    result[x][y] = v1[x][y] * val;
	}
    }
    return(result);
}

fn get_data(v1: &mut Vec<Vec<f64>>)
{
    let mut count = 1.0;
    for x in 0..v1.len(){
	for y in 0..v1[0].len(){
	    v1[x][y] = rand::random();
	    v1[x][y] -= 0.5;
	    v1[x][y] = v1[x][y]*3.2;
	    //v1[x][y] = count;
	    count+=1.0;
	}
    }
}


fn rectified_linear_unit(v1: &mut Vec<Vec<f64>>) -> Vec<Vec<f64>>
{
    let mut result = vec![vec![0.0; v1[0].len()]; v1.len()];
    
    for x in 0..v1.len(){
	for y in 0..v1[0].len(){
	    result[x][y] = if v1[x][y] < 0.0 { 0.0 } else { v1[x][y] };  
	}
    }
    return result;
}
