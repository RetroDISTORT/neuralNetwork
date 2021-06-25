/*
Create a new sample project:
cargo new test_project && cd test_project && cargo build && cargo run

Run the project:
cargo run
*/

extern crate rand; //Used for rand
use rand::distributions::{Normal};

const INPUT_NODES: usize = 2;
const OUTPUT_NODES: usize = 2;
const HIDDEN_NODES: usize = 3;
const BATCH_SIZE: usize = 8;

fn main(){
    //let args: Vec<String> = env::args().collect(); //[][inputs][sub][output]
    let mut inputData = vec![vec![0.0; INPUT_NODES]; BATCH_SIZE];
    let mut weightMatrix = vec![vec![0.0; HIDDEN_NODES]; INPUT_NODES];
    
    get_data(&mut inputData);
    get_data(&mut weightMatrix);
    //println!("{:?}",dot_product(&mut inputData, &mut weightMatrix));
    println!("{:?}",transpose(&mut inputData));

    
    //println!("{:?}", inputData);
    //println!("{:?}", weightMatrix);
    //println!("{:?}", weightMatrix.len());
    
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

fn transpose(v1: &mut Vec<Vec<f64>>) -> Vec<Vec<f64>>
{
    let mut result = vec![vec![0.0; v1.len()]; v1[0].len()];
    let mut count = 1.0;
    
    for x in 0..result.len(){
	for y in 0..result[0].len(){
	    result[x][y] = v1[y][x];
	}
    }
    return(result);
}

fn get_data(v1: &mut Vec<Vec<f64>>)
{
    let mut count = 1.0;
    for x in 0..v1.len(){
	for y in 0..v1[0].len(){
	    v1[x][y] = count;//rand::random();
	    count+=1.0;
	}
    }
}

fn rectified_linear_unit(v1: &mut Vec<Vec<f64>>)
{
    for x in 0..v1.len(){
	for y in 0..v1[0].len(){
	    if (v1[x][y] < 0.0)
	    {
		v1[x][y] = 0.0;
	    }
	}
    }
}
