use pyrus_nn::network::Sequential;
use serde_derive::Deserialize;

#[derive(Deserialize)]
pub struct Payload {
    pub x: Vec<Vec<f32>>,
    pub y: Option<Vec<Vec<f32>>>,
    pub model: Sequential,
}
