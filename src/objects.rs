use serde_derive::{Deserialize};

#[derive(Deserialize)]
pub struct Payload {
    pub X: Vec<Vec<f32>>,
    pub y: Vec<Vec<f32>>
}