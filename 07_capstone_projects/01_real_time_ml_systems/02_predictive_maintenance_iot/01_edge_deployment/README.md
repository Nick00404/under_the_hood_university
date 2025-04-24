## 📘 **Capstone: Edge Deployment – Structured Index**

---

### 🧩 **01. Deploying ML Models on Edge Devices**

#### 📌 **Includes: `edge_device_deployment_guide.md`, `edge_deployment_tflite.ipynb`**

##### **Subtopics:**
- **What Is Edge Deployment?**
  - Use cases: offline inference, privacy, reduced latency
- **Supported Hardware**
  - Raspberry Pi, Android devices, Coral TPU, microcontrollers
- **TensorFlow Lite (TFLite) Overview**
  - Lightweight format for on-device inference
- **End-to-End Deployment Guide**
  - Installing dependencies, deploying models, running test inference
- **Example:** Running a digit recognition or object detection model on a Raspberry Pi using TFLite

---

### 🧩 **02. Model Optimization and Compression Techniques**

#### 📌 **Includes: `model_compression.ipynb`**

##### **Subtopics:**
- **Why Compress Models for Edge?**
  - Reduce memory, power, and inference latency
- **Pruning and Weight Clustering**
  - Techniques to eliminate redundancy while retaining accuracy
- **Knowledge Distillation**
  - Train a smaller model to mimic a larger one
- **Model Size vs Accuracy Tradeoffs**
  - How much compression is “too much”?
- **Example:** Applying pruning and visualizing model size reduction

---

### 🧩 **03. Quantization for Efficient Inference**

#### 📌 **Includes: `quantization_tflite.ipynb`**

##### **Subtopics:**
- **What Is Quantization?**
  - Reducing model precision (e.g., float32 → int8)
- **Types of Quantization in TFLite**
  - Post-training dynamic, full integer, and quantization-aware training
- **Performance Gains and Accuracy Drops**
  - Benchmarking before/after quantization
- **Deployment Compatibility**
  - Ensuring quantized models run on edge hardware
- **Example:** Converting a model to int8 and benchmarking latency improvement on an edge device

---


✅ **Fantastic!** Starting your second capstone:

# 🌐 **Predictive Maintenance IoT: Edge Deployment**

---

## 📁 **Folder Structure**
```
📂 01_real_time_ml_systems
└── 📂 02_predictive_maintenance_iot
    └── 📂 01_edge_deployment
        ├── 📒 edge_deployment_tflite.ipynb
        ├── 📄 edge_device_deployment_guide.md
        ├── 📒 model_compression.ipynb
        └── 📒 quantization_tflite.ipynb
```

---

## 🎯 **Capstone Overview**

You’ll build a lightweight ML system deployed directly on edge devices (e.g., Raspberry Pi, Arduino):

- ✅ **TinyML models** (TensorFlow Lite)
- ✅ **Model Compression** (pruning, quantization)
- ✅ **Device-Specific Deployment**

---

## 📖 **Learning Goals**

- Create ML models suitable for resource-constrained devices.
- Compress models using pruning & quantization.
- Deploy models directly onto IoT devices (e.g., Raspberry Pi).

---

# 📒 **Lab Notebooks (Step-by-Step)**

---

## 🟢 **1. `edge_deployment_tflite.ipynb`**
- **Train simple ML model** (sensor data classification).
- **Convert model to TensorFlow Lite** (TFLite).
- **Deploy & test locally** using TFLite interpreter.

---

## 📗 **2. `model_compression.ipynb`**
- **Apply pruning & weight clustering** techniques.
- **Evaluate accuracy vs. model size tradeoffs**.
- **Prepare ultra-compact model** for tiny hardware.

---

## 📘 **3. `quantization_tflite.ipynb`**
- **Perform INT8 quantization** on trained TFLite models.
- **Benchmark inference speed & memory usage**.
- **Ensure minimal accuracy loss post-quantization**.

---

## 📄 **4. `edge_device_deployment_guide.md`**
- Step-by-step instructions for deploying TFLite models onto edge hardware (e.g., Raspberry Pi).
- Guidance on setup, troubleshooting, and best practices.

---

## 🚀 **Skills You'll Develop**

- IoT ML model deployment & optimization.
- Model quantization & compression techniques.
- Real-world skills for resource-constrained ML.

---

## 🎯 **Next Step**

Ready to kick off your first notebook?

**Let's start with** 📒 **`edge_deployment_tflite.ipynb`**?

✅ **Awesome!** Let’s begin your first notebook:

# 📒 `edge_deployment_tflite.ipynb`
## 📁 `02_predictive_maintenance_iot/01_edge_deployment`

---

## 🎯 **Learning Goals**

- Build & train a simple ML model for predictive maintenance (sensor classification).
- Convert TensorFlow model to TensorFlow Lite (TFLite) for edge deployment.
- Test inference with TFLite locally (ready for Raspberry Pi/Arduino deployment).

---

## 💻 **Runtime Setup**

| Component        | Setup                        |
|------------------|------------------------------|
| Framework        | TensorFlow, TFLite ✅        |
| Simulation Data  | Sensor data (simulated) ✅   |
| Platform         | Colab-compatible ✅          |

---

## 🚧 **1. Install Dependencies**

```bash
!pip install tensorflow numpy
```

---

## 📡 **2. Simulate Sensor Data**

Generate dummy sensor data (for predictive maintenance):

```python
import numpy as np
import tensorflow as tf

# Simulate sensor data
np.random.seed(42)
num_samples = 1000
num_features = 10

# Features (sensor signals)
X = np.random.rand(num_samples, num_features).astype(np.float32)

# Labels (0: normal, 1: faulty)
y = (np.sum(X, axis=1) > 5).astype(np.float32)
```

---

## 🧠 **3. Train a Simple Model**

```python
# Simple sequential model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(num_features,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X, y, epochs=10, batch_size=32)
```

---

## 🚀 **4. Convert to TFLite Model**

```python
# Convert model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

---

## 🖥️ **5. Test TFLite Model Locally**

```python
# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

# Input & Output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test inference on sample data
test_sample = np.array([X[0]], dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], test_sample)
interpreter.invoke()

# Get prediction
prediction = interpreter.get_tensor(output_details[0]['index'])
print("TFLite prediction:", prediction)
```

---

## 📌 **What You Learned**

- Created a **TensorFlow Lite model** suitable for edge devices.
- Successfully trained, converted, and tested the TFLite model.
- Prepared groundwork for IoT predictive maintenance deployments.

---

## ✅ **Wrap-Up**

| Milestone                                   | ✅ |
|---------------------------------------------|----|
| Sensor data simulated                       | ✅ |
| Model trained and validated                 | ✅ |
| TFLite conversion successful                | ✅ |
| TFLite inference tested locally             | ✅ |

---

## 🎯 **Next Step**

Next notebook:  
⚡ **`model_compression.ipynb`**

We'll reduce the model size further for ultra-lightweight deployment.

**Ready for the next step?**

✅ **Great!** Time to compress your model even further:

# ⚡ `model_compression.ipynb`  
## 📁 `02_predictive_maintenance_iot/01_edge_deployment`

---

## 🎯 **Learning Goals**

- Apply model compression techniques (pruning, weight clustering).
- Reduce model size significantly for IoT edge deployments.
- Evaluate performance trade-offs (accuracy vs. size).

---

## 💻 **Runtime Setup**

| Component              | Setup                      |
|------------------------|----------------------------|
| Framework              | TensorFlow Model Optimization Toolkit ✅ |
| Evaluation             | Accuracy and Size Metrics ✅|
| Platform               | Colab-friendly ✅          |

---

## 🚧 **1. Install Dependencies**

```bash
!pip install tensorflow tensorflow-model-optimization numpy
```

---

## 📂 **2. Reload Original Model**

(Quick retrain for continuity)

```python
import numpy as np
import tensorflow as tf

# Re-generate sensor data
np.random.seed(42)
X = np.random.rand(1000, 10).astype(np.float32)
y = (np.sum(X, axis=1) > 5).astype(np.float32)

# Retrain original model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=5, batch_size=32)
```

---

## 🔪 **3. Apply Pruning (Compression)**

```python
import tensorflow_model_optimization as tfmot

pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(0.7, 0)
}

pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

# Compile & retrain pruned model
pruned_model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])

pruned_model.fit(X, y, epochs=5, batch_size=32,
                 callbacks=[tfmot.sparsity.keras.UpdatePruningStep()])

# Strip pruning wrappers to finalize
final_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
```

---

## 📏 **4. Compare Model Size Before vs. After**

```python
def get_gzipped_model_size(model):
    import os
    import tempfile
    import zipfile

    _, temp_path = tempfile.mkstemp('.h5')
    model.save(temp_path, include_optimizer=False)

    zip_path = temp_path + '.zip'
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(temp_path)

    size = os.path.getsize(zip_path) / (1024 ** 2)
    print(f"Model size: {size:.2f} MB")
    return size

print("Original Model:")
get_gzipped_model_size(model)

print("Pruned Model:")
get_gzipped_model_size(final_model)
```

---

## 🧪 **5. Evaluate Accuracy Post-Compression**

```python
# Evaluate pruned model
loss, acc = final_model.evaluate(X, y, verbose=0)
print(f"Accuracy after pruning: {acc:.4f}")
```

---

## 💾 **6. Convert Compressed Model to TFLite**

```python
converter = tf.lite.TFLiteConverter.from_keras_model(final_model)
tflite_model = converter.convert()

# Save compressed TFLite model
with open('compressed_model.tflite', 'wb') as f:
    f.write(tflite_model)
```

---

## 🧠 **What You Learned**

- Compressed your model significantly using pruning (70% weights removed).
- Evaluated accuracy vs. size tradeoffs for deployment.
- Generated ultra-lightweight TFLite models for edge IoT devices.

---

## ✅ **Wrap-Up**

| Milestone                           | ✅ |
|-------------------------------------|----|
| Original model compressed via pruning | ✅ |
| Evaluated size and accuracy         | ✅ |
| Final compressed TFLite model       | ✅ |

---

## 🎯 **Next Step**

Next notebook:  
📦 **`quantization_tflite.ipynb`**

We'll further compress by applying INT8 quantization for maximum efficiency.

**Ready to quantize your model for ultimate IoT readiness?**

✅ **Excellent!** Let's take your compressed model to the ultimate IoT-ready level:

# 📦 `quantization_tflite.ipynb`  
## 📁 `02_predictive_maintenance_iot/01_edge_deployment`

---

## 🎯 **Learning Goals**

- Apply INT8 quantization to significantly reduce model size.
- Benchmark inference speed and memory usage.
- Evaluate the accuracy trade-off of quantization.

---

## 💻 **Runtime Setup**

| Component      | Setup                             |
|----------------|-----------------------------------|
| Framework      | TensorFlow, TFLite Quantization ✅ |
| Metrics        | Size, Latency, Accuracy ✅        |
| Platform       | Colab-friendly ✅                 |

---

## 🚧 **1. Install Dependencies**

```bash
!pip install tensorflow numpy
```

---

## 📂 **2. Reload Compressed Model**

Recreate quickly for continuity:

```python
import numpy as np
import tensorflow as tf

# Data
np.random.seed(42)
X = np.random.rand(1000, 10).astype(np.float32)
y = (np.sum(X, axis=1) > 5).astype(np.float32)

# Rebuild compressed model (quickly)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=5, batch_size=32)
```

---

## ⚡ **3. Apply INT8 Quantization**

```python
def representative_data_gen():
    for i in range(100):
        yield [X[i:i+1]]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_types = [tf.int8]

# INT8 quantization
tflite_quant_model = converter.convert()

# Save quantized model
with open('int8_quantized_model.tflite', 'wb') as f:
    f.write(tflite_quant_model)
```

---

## 📏 **4. Measure Model Size Reduction**

```python
import os

size = os.path.getsize('int8_quantized_model.tflite') / 1024
print(f"INT8 quantized model size: {size:.2f} KB")
```

---

## ⏱️ **5. Benchmark Quantized Model Speed**

```python
import time

interpreter = tf.lite.Interpreter(model_path='int8_quantized_model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

test_input = np.array([X[0]], dtype=np.float32)

start_time = time.time()
for _ in range(1000):
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
end_time = time.time()

latency = (end_time - start_time) / 1000
print(f"Avg latency per inference: {latency * 1000:.2f} ms")
```

---

## 🧪 **6. Evaluate Accuracy Post-Quantization**

```python
correct = 0
for i in range(100):
    interpreter.set_tensor(input_details[0]['index'], X[i:i+1])
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])[0][0]
    pred_label = 1 if pred > 0.5 else 0
    if pred_label == y[i]:
        correct += 1

accuracy = correct / 100
print(f"Quantized model accuracy: {accuracy:.2%}")
```

---

## 🧠 **What You Learned**

- Achieved dramatic model size reduction using INT8 quantization.
- Benchmarked inference speed suitable for real-time IoT applications.
- Maintained acceptable accuracy after aggressive quantization.

---

## ✅ **Wrap-Up**

| Milestone                          | ✅ |
|------------------------------------|----|
| INT8 quantization applied          | ✅ |
| Size & latency benchmarks completed| ✅ |
| Accuracy evaluated and validated   | ✅ |

---

🎉 **Capstone Completion!** 🎉  
You've successfully prepared an **ultra-efficient ML model** for IoT edge deployment:

- Built ✅
- Compressed ✅
- Quantized ✅

## 🎯 **Next Step**

Next:  
📄 **`edge_device_deployment_guide.md`**

We'll finalize deployment instructions for hardware like **Raspberry Pi**.

**Ready for the final deployment guide?**

✅ **Here's your detailed guide ready for deployment:**

# 📄 **`edge_device_deployment_guide.md`**
## 📁 `02_predictive_maintenance_iot/01_edge_deployment`

---

# 🚀 **Edge Device Deployment Guide**

This guide explains how to deploy your optimized **TensorFlow Lite (TFLite)** model onto an edge device such as a **Raspberry Pi**.

---

## 📋 **Prerequisites**

- Raspberry Pi (e.g., Raspberry Pi 4)
- MicroSD card with Raspberry Pi OS installed
- SSH access or direct console access
- Python installed (`Python 3.7+`)
- Internet connection

---

## ⚙️ **1. Setting up the Raspberry Pi**

First, update the Raspberry Pi OS:

```bash
sudo apt update
sudo apt upgrade -y
```

---

## 🐍 **2. Install Python Dependencies**

Install Python packages required for TFLite inference:

```bash
sudo apt install python3-pip
pip3 install numpy tensorflow-lite
```

---

## 📂 **3. Transfer Your Model**

Copy the quantized `.tflite` model (`int8_quantized_model.tflite`) to the Raspberry Pi using SCP or USB.

**Using SCP (recommended)**:

```bash
scp int8_quantized_model.tflite pi@raspberrypi.local:~/model.tflite
```

---

## 📟 **4. Create Inference Script**

Create a file named `predict.py` on Raspberry Pi:

```python
import numpy as np
import tflite_runtime.interpreter as tflite
import time

# Load model
interpreter = tflite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Example sensor input
sensor_input = np.random.rand(1, 10).astype(np.float32)

# Run inference
interpreter.set_tensor(input_details[0]['index'], sensor_input)

start = time.time()
interpreter.invoke()
end = time.time()

# Prediction
output = interpreter.get_tensor(output_details[0]['index'])[0][0]
print(f"Prediction: {output:.4f}")
print(f"Inference time: {(end - start) * 1000:.2f} ms")
```

---

## 🚀 **5. Run Your Model on Raspberry Pi**

```bash
python3 predict.py
```

You should see a prediction result along with inference time:

```
Prediction: 0.8321
Inference time: 1.45 ms
```

---

## 🛠️ **6. Troubleshooting Tips**

- **Missing packages**:
```bash
pip3 install --upgrade numpy tflite-runtime
```

- **Model path issues**:
  - Ensure `model.tflite` is in the same directory as your Python script.
  - Use absolute paths if necessary (`/home/pi/model.tflite`).

---

## 📌 **7. Best Practices**

- Always test your model locally first (e.g., Colab), then deploy.
- Quantized models run faster on Raspberry Pi.
- Monitor Raspberry Pi resources (`htop`) during inference.

---

## 📈 **8. Next Steps for Production**

- Integrate with sensor hardware via GPIO.
- Set up automatic script execution on boot.
- Send predictions to cloud services (MQTT, REST APIs).

---

🎯 **Congratulations!**  
You’ve successfully deployed a **fully optimized ML model** on your edge device, ready for real-world predictive maintenance.

---

✅ **Guide Wrap-Up**

| Task                                | ✅ |
|-------------------------------------|----|
| Raspberry Pi setup                  | ✅ |
| Model transferred and tested        | ✅ |
| Inference speed verified            | ✅ |
| Deployment best practices covered   | ✅ |

---

🌟 **Ready for your next exciting ML challenge?** 🌟  
Just let me know, Professor!



















