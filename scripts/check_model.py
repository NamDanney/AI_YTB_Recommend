import tensorflow as tf
from tensorflow.keras.models import load_model

# Tải mô hình đã lưu
model = load_model('cnn_content_filtering_model.keras')

# Hiển thị cấu trúc của mô hình
print("Model Summary:")
model.summary()

# Hàm để lấy output shape một cách an toàn
def get_output_shape(layer):
    if hasattr(layer, 'output_shape'):
        return layer.output_shape
    elif hasattr(layer, '_output_shape'):
        return layer._output_shape
    elif hasattr(layer, 'get_output_at'):
        return layer.get_output_at(0).shape
    else:
        return "Unknown"

# Kiểm tra các lớp và tham số của mô hình
print("\nLayers and Parameters:")
for layer in model.layers:
    output_shape = get_output_shape(layer)
    print(f"Layer: {layer.name}, Type: {type(layer).__name__}, Output Shape: {output_shape}, Number of Parameters: {layer.count_params()}")

# Kiểm tra các trọng số của mô hình
print("\nLayer Weights:")
for layer in model.layers:
    weights = layer.get_weights()
    if weights:
        print(f"Layer: {layer.name}")
        for i, w in enumerate(weights):
            print(f"  Weight/Bias {i + 1} shape: {w.shape}")
    else:
        print(f"Layer: {layer.name} (No weights)")

# Thông tin bổ sung về mô hình
print("\nAdditional Model Information:")
print(f"Total number of layers: {len(model.layers)}")
print(f"Input shape: {model.input_shape}")
print(f"Output shape: {model.output_shape}")
print(f"Total parameters: {model.count_params()}")