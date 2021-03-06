use lambda_http::{http::Method, lambda, IntoResponse, Request, RequestExt};
use lambda_runtime::{error::HandlerError, Context};
use ndarray::{Array1, Array2};
use pyrus_nn::{
    costs::CostFunc,
    activations::Activation,
    layers::{Dense, Layer},
    network::Sequential,
};
use serde_json::{json, Value};

mod objects;
use crate::objects::Payload;

fn main() {
    lambda!(handler);
}

fn handler(request: Request, _: Context) -> Result<impl IntoResponse, HandlerError> {

    match request.method() {
        &Method::GET => get_model(&request),
        &Method::POST => train_or_predict(&request),
        _ => Ok(json!({"message": "Only POST and GET supported."})),
    }
}

// Here the request is expect to contain 'X' in body
fn train_or_predict(request: &Request) -> Result<Value, HandlerError> {
    match request
        .payload::<Payload>()
        .map_err(|_e| HandlerError::from("Failed to parse payload"))?
    {
        Some(payload) => {
            // Will always have x and model
            let x_array = array_from_nested_vec(payload.x);
            let mut model = payload.model;

            // Presence of y determines if we're training or predicting
            match payload.y {
                Some(target) => {
                    let y_array = array_from_nested_vec(target);
                    model.fit(x_array.view(), y_array.view());
                    Ok(json!({"message": "Training round completed!", "model": model}))
                }
                None => {
                    let out = model
                        .predict(x_array.view())
                        .outer_iter()
                        .map(|v| v.to_vec())
                        .collect::<Vec<Vec<f32>>>();

                    Ok(json!({"message": "Created predictions!", "predictions": out}))
                }
            }
        }

        None => Err(HandlerError::from("No payload supplied!")),
    }
}

fn array_from_nested_vec(vec: Vec<Vec<f32>>) -> Array2<f32> {
    let shape = (vec.len(), vec[0].len());
    Array1::from_iter(vec.into_iter().flat_map(|v| v))
        .into_shape(shape)
        .unwrap()
}

// Request is expected to contain 'X' array in query string params
fn get_model(_request: &Request) -> Result<Value, HandlerError> {
    Ok(json!({"message": "New model, just for you!", "model": build_base_model()}))
}

// The base/default model; TODO: Support parametrization
fn build_base_model() -> Sequential {
    let mut nn = Sequential::new();
    nn.add(Dense::new(4, 6, Activation::Tanh)).unwrap();
    nn.add(Dense::new(6, 6, Activation::Tanh)).unwrap();
    nn.add(Dense::new(6, 3, Activation::Softmax)).unwrap();
    nn.n_epoch = 1;
    nn.cost = CostFunc::CrossEntropy;
    nn
}

#[cfg(test)]
mod tests {
    use super::*;
    use lambda_http::http::StatusCode;

    #[test]
    fn handler_handles() {
        let request = Request::default();

        let response = handler(request, Context::default())
            .expect("expected Ok(_) value")
            .into_response();
        assert_eq!(response.status(), StatusCode::OK)
    }
}
