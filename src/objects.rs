use serde_derive::{Deserialize};
use pyrus_nn::network::Sequential;


#[derive(Deserialize)]
pub struct Payload {
    pub x: Vec<Vec<f32>>,
    pub y: Option<Vec<Vec<f32>>>,
    pub model: Sequential
}
