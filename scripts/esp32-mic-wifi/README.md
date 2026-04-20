# ESP32 + INMP441 — WiFi 实时麦克风流

## 一次性环境准备

1. 在 VSCode 扩展市场安装 **PlatformIO IDE**
2. 在 `src/main.cpp` 顶部填入你的 WiFi SSID 和密码
3. 用 USB 连接 ESP32

## 构建 / 上传

在 VSCode 打开本文件夹（`esp32-mic-wifi`），PlatformIO 会自动：
- 下载 `espressif32` 平台（首次约 200 MB）
- 下载 `ESPAsyncWebServer` / `AsyncTCP` 库

然后点击底部工具栏：
- ✓ 编译
- → 上传
- 🔌 串口监视器（看 ESP32 获取到的 IP 地址）

## 使用

浏览器打开 `http://<ESP32的IP>/` 即可看到实时波形图。
原始数据流：`ws://<ESP32的IP>/ws`（每次消息是一个数字字符串，平均振幅）。

## 接线（与原代码一致）

| INMP441 | ESP32 |
|---------|-------|
| VDD     | 3V3   |
| GND     | GND   |
| L/R     | GND   |
| WS      | GPIO25|
| SCK     | GPIO33|
| SD      | GPIO32|
