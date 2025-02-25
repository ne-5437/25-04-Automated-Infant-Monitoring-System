#include <TensorFlowLite.h>
#include "model_data.h"  // Include the generated model data

// Set up the TensorFlow Lite interpreter
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;

const tflite::Model* model = tflite::GetModel(g_model);
if (model->version() != TFLITE_SCHEMA_VERSION) {
  error_reporter->Report("Model provided is schema version %d not equal to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
  return;
}

// Set up the interpreter
tflite::MicroInterpreter interpreter(model, tensor_arena, kTensorArenaSize, error_reporter);
interpreter.AllocateTensors();

// Get input and output tensors
TfLiteTensor* input = interpreter.input(0);
TfLiteTensor* output = interpreter.output(0);

void setup() {
  Serial.begin(115200);

  // Example input data (replace with actual sensor data)
  float input_data[128 * 128 * 3] = {0};  // Adjust based on your model's input shape
  for (int i = 0; i < 128 * 128 * 3; i++) {
    input_data[i] = 0.5f;  // Example normalized input
  }

  // Copy input data to the model's input tensor
  for (int i = 0; i < 128 * 128 * 3; i++) {
    input->data.f[i] = input_data[i];
  }

  // Run inference
  if (interpreter.Invoke() != kTfLiteOk) {
    error_reporter->Report("Inference failed");
    return;
  }

  // Get the output
  float normal_prob = output->data.f[0];
  float low_fever_prob = output->data.f[1];
  float high_fever_prob = output->data.f[2];

  // Print the results
  Serial.print("Normal Probability: ");
  Serial.println(normal_prob);
  Serial.print("Low Fever Probability: ");
  Serial.println(low_fever_prob);
  Serial.print("High Fever Probability: ");
  Serial.println(high_fever_prob);
}

void loop() {
  // Empty loop
}