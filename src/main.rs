use lambda_http::{lambda, IntoResponse, Request, RequestExt, http::Method};
use lambda_http::request::RequestContext;
use lambda_runtime::{error::HandlerError, Context};
use serde_json::{json, Value};

fn main() {
    lambda!(handler)
}

fn handler(request: Request, _: Context) -> Result<impl IntoResponse, HandlerError> {

    // POST we're training GET we're predicting
    match request.method() {
        &Method::GET => handle_predicting(&request),
        &Method::POST => handle_training(&request),
        _ => Ok(json!({"message": "Only POST and GET supported."}))
    }

}

fn handle_training(request: &Request) -> Result<Value, HandlerError> {
    Ok(json!({"message": "Training round completed!"}))
}

fn handle_predicting(request: &Request) -> Result<Value, HandlerError> {
    Ok(json!({"message": "Training round completed!"}))
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