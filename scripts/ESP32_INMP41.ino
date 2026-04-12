#include <driver/i2s.h>


// I2S pins
#define I2S_WS  25   // Word-Select (LRCL)
#define I2S_SCK 33   // Serial-Clock (BCLK)
#define I2S_SD  32   // Serial-Data (DIN)


// I2S configuration
#define I2S_SAMPLE_RATE 16000   // 16 kHz sample rate
#define I2S_BUFFER_SIZE 1024    // Buffer size


void setup() {
  Serial.begin(115200);
  Serial.println("INMP441 Microphone Test - Serial Plotter");


  // Configure I2S
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),   // Master, RX mode
    .sample_rate = I2S_SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,          // 32-bit data
    .channel_format = I2S_CHANNEL_FMT_ONLY_RIGHT,           // Mono (left channel)
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,              // Interrupt level 1
    .dma_buf_count = 4,                                    // Number of DMA buffers
    .dma_buf_len = I2S_BUFFER_SIZE,                        // Buffer length
    .use_apll = false
  };


  // Install I2S driver
  i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);


  // Set I2S pins
  i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = -1,   // Not used
    .data_in_num = I2S_SD
  };


  i2s_set_pin(I2S_NUM_0, &pin_config);
}


void loop() {
  int32_t audio_samples[I2S_BUFFER_SIZE];
  size_t bytes_read;


  // Read audio data from I2S
  i2s_read(I2S_NUM_0, audio_samples, sizeof(audio_samples), &bytes_read, portMAX_DELAY);


  // Calculate average amplitude
  int64_t sum = 0;
  for (int i = 0; i < bytes_read / sizeof(int32_t); i++) {
    sum += abs(audio_samples[i]);   // Use absolute value to measure amplitude
  }


  int32_t average_amplitude = sum / (bytes_read / sizeof(int32_t));


  // Send average amplitude to Serial Plotter
  Serial.println(average_amplitude);


  //delay(10);   // Add a small delay to control the data rate
}