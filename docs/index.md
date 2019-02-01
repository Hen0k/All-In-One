![SingularityNet.io](./assets/singnet-logo.jpg?raw=true 'SingularityNET')

[![CircleCI](https://circleci.com/gh/IsraelAbebe/All-In-One.svg?style=svg)](https://circleci.com/gh/IsraelAbebe/All-In-One)

# All-In-One
All in one [paper](https://arxiv.org/abs/1611.00851) implementation

## Welcome


All in one convolutional network for face analysis presents a multipurpose algorithm for simultaneous face detection, face alignment, pose estimation, gender recognition, smile detection, age estimation and face recognition using a single convolutional neural network(CNN).


## How does it work?

The user must provide a request satisfying the proto descriptions [given](../Service/service_spec/all_in_one.proto). That is

* An request with `image_type`: the type of the input image. 
* And the image `image`: the string64 encoded input image.

The following options are available for image type: `png`, `jpg`

The input image can be `monochrome`, `rgb`, `rgba`. Additional values hadn't yet been tested.

### Using the service on the platform

The returned result has the following form: 
```bash
message All_In_One_Request {
    string image = 1;
    string image_type = 2;
}
```

An example result obtained after passing the [image](../imgs/adele.png)
```bash
bounding_boxes {
  x: 62
  y: 211
  w: 194
  h: 344
}
age: 0
smile: "False"
gender: "Female"

```
