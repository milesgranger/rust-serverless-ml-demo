
use lambda_http::{lambda, IntoResponse, Request, RequestExt, http::Method};
use lambda_http::request::RequestContext;
use lambda_runtime::{error::HandlerError, Context};


use serde_json::{json, Value};
use ndarray::{arr2, Array2, stack, Array1, Axis};
use failure::bail;


mod objects;
use objects::Payload;

fn main() {
    lambda!(handler)
}

fn handler(request: Request, _: Context) -> Result<impl IntoResponse, HandlerError> {

    // POST we're training, GET we're predicting
    match request.method() {
        &Method::GET => handle_predicting(&request),
        &Method::POST => handle_training(&request),
        _ => Ok(json!({"message": "Only POST and GET supported."}))
    }
}




// Here the request is expect to contain 'X' in body
fn handle_training(request: &Request) -> Result<Value, HandlerError> {

    match request.payload::<Payload>()
        .map_err(|_e| HandlerError::from("Failed to parse payload"))? {

        Some(payload) => {

            let x_array = stack!(Axis(0), payload.X.iter().map(|v| v.as_slice()).collect::<Vec<&[f32]>>().as_slice());
            let y_array = stack!(Axis(0), payload.y.iter().map(|v| v.as_slice()).collect::<Vec<&[f32]>>().as_slice());

            Ok(json!({"message": "Training round completed!"}))
        },
        None => Err(HandlerError::from("No training data supplied in X!"))
    }


}

// Request is expected to contain 'X' array in query string params
fn handle_predicting(request: &Request) -> Result<Value, HandlerError> {
    match request.path_parameters().get("name") {
        Some(name) => Ok(json!({"message": format!("Predicting using model: '{}' completed!", name)})),
        None => Ok(json!({"message": "Could not predict for unknown model, provide a model name in path"}))
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handler_handles() {
        let request = Request::default();
        let expected = json!({
        "message": "Go Serverless v1.0! Your function executed successfully!"
        })
        .into_response();
        let response = handler(request, Context::default())
            .expect("expected Ok(_) value")
            .into_response();
        assert_eq!(response.body(), expected.body())
    }
}