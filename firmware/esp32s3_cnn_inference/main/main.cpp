// On-device inference harness for the reference-conditioned pill CNN.
//
// Loads the int8 TFLite model (model_data.h, produced by
// convert_cnn_to_tflite.py), runs it on the ESP32-S3, and reports latency,
// tensor-arena usage and the prediction over the serial console. This proves
// the 80k-param CNN runs on the ESP32-S3 and measures it — the Phase-2 goal of
// "how many models can we run on the board".
//
// It classifies a SINGLE 6-channel cell (cell RGB + empty-reference RGB,
// 128x96). Locating and cropping the 21 cells from a full frame still needs
// the OpenCV front-end that runs off-board; wiring the camera capture into a
// per-cell buffer is the next step (see README). Here the input is a fixed
// test pattern so the timing/memory numbers are reproducible.
//
// Reference impl: written against the esp-tflite-micro API. Validate on real
// hardware — arena size and op set may need tuning after the model is regen'd.

#include <cmath>
#include <cstdio>

#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"

namespace {
// The CNN's peak activation (16ch x 128 x 96 int8 after the first conv) plus
// weights fits well under this; 512 KB in PSRAM leaves generous headroom.
constexpr int kArenaBytes = 512 * 1024;
constexpr int kInputElems = 6 * 128 * 96;  // NHWC: 128 x 96 x 6
constexpr int kWarmup = 5;
constexpr int kRuns = 50;

uint8_t* g_arena = nullptr;
}  // namespace

extern "C" void app_main(void) {
  tflite::InitializeTarget();

  const tflite::Model* model = tflite::GetModel(g_pill_cnn_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    printf("model schema %lu != supported %d — regenerate model_data.h\n",
           (unsigned long)model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  g_arena = static_cast<uint8_t*>(
      heap_caps_malloc(kArenaBytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  if (g_arena == nullptr) {
    printf("failed to allocate %d KB tensor arena in PSRAM — is PSRAM enabled?\n",
           kArenaBytes / 1024);
    return;
  }

  // Register exactly the ops this CNN uses (Conv/MaxPool/ReLU/GAP/FC/softmax +
  // the quantize/dequantize wrappers the int8 graph adds). GlobalAveragePool
  // maps to Mean or AveragePool2D depending on the converter, so both are here.
  static tflite::MicroMutableOpResolver<11> resolver;
  resolver.AddConv2D();
  resolver.AddMaxPool2D();
  resolver.AddRelu();
  resolver.AddMean();
  resolver.AddAveragePool2D();
  resolver.AddReshape();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddQuantize();
  resolver.AddDequantize();

  static tflite::MicroInterpreter interpreter(model, resolver, g_arena,
                                              kArenaBytes);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    printf("AllocateTensors failed — arena too small or an op is unregistered\n");
    return;
  }

  TfLiteTensor* input = interpreter.input(0);
  TfLiteTensor* output = interpreter.output(0);
  if (input->type != kTfLiteInt8) {
    printf("expected int8 input; got type %d — reconvert with int8 io\n",
           input->type);
    return;
  }

  // Fill with a mid-gray test pattern (0.5 in [0,1]), quantized to the input's
  // int8 scale/zero-point. Replace this with a real captured cell later.
  const float kTestValue = 0.5f;
  const int8_t q = static_cast<int8_t>(
      std::lround(kTestValue / input->params.scale) + input->params.zero_point);
  for (int i = 0; i < kInputElems; ++i) input->data.int8[i] = q;

  for (int i = 0; i < kWarmup; ++i) interpreter.Invoke();

  const int64_t t0 = esp_timer_get_time();
  for (int i = 0; i < kRuns; ++i) {
    if (interpreter.Invoke() != kTfLiteOk) {
      printf("Invoke failed at run %d\n", i);
      return;
    }
  }
  const int64_t t1 = esp_timer_get_time();
  const float ms_per = (t1 - t0) / 1000.0f / kRuns;

  // Dequantize the two logits and take argmax (0=empty, 1=pill).
  float logit[2];
  for (int c = 0; c < 2; ++c) {
    logit[c] = (output->data.int8[c] - output->params.zero_point) *
               output->params.scale;
  }
  const int pred = logit[1] > logit[0] ? 1 : 0;

  printf("\n=== PillWatch CNN on ESP32-S3 ===\n");
  printf("model:        %u bytes int8\n", (unsigned)g_pill_cnn_model_len);
  printf("latency:      %.2f ms / inference (avg of %d)\n", ms_per, kRuns);
  printf("throughput:   %.1f inferences/s\n", 1000.0f / ms_per);
  printf("arena used:   %d KB of %d KB\n",
         (int)(interpreter.arena_used_bytes() / 1024), kArenaBytes / 1024);
  printf("PSRAM free:   %d KB\n",
         (int)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024));
  printf("prediction:   %s (logits empty=%.3f pill=%.3f)\n",
         pred ? "PILL" : "EMPTY", logit[0], logit[1]);
  printf("(test-pattern input — wire in a real cell crop next)\n");

  while (true) vTaskDelay(pdMS_TO_TICKS(1000));
}
