# rust-ml-demo

---

[![Build Status](https://travis-ci.com/milesgranger/rust-serverless-ml-demo.svg?branch=master)](https://travis-ci.com/milesgranger/rust-serverless-ml-demo)


Example of running / training a neural network through AWS Lambda. 

It uses a [pyrus-nn](https://github.com/milesgranger/pyrus-nn), a micro neural network lib I've
written in Rust which allows for primitive serialization of deep learning models. (`.json`, `.yaml`, etc.)

---

Flow:
  - Client makes GET request to the endpoint
  - Client receives a JSON representation of a neural network
  - Client can then take that model and do the following:
    - POST request with `model`, `x`, `y` -> Lambda function will train the model, 
      and returns the new model with updated weights in the response.
    - POST request with `model` and `x` -> Lambda function will predict on `x` using the model

---

Requires:  
  - [NPX](https://www.npmjs.com/package/npx) cli
  - [Rust >= 1.31](https://rustup.rs/)
  - `serverless-rust` plugin -> `npm i -D serverless-rust`
     - [more info here](https://github.com/softprops/serverless-rust)
  - `npx serverless deploy` after configuring the `serverless.yml` file
  

