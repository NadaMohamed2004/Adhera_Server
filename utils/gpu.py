import torch
import tensorflow as tf

def get_device():
    # PyTorch GPU check
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # TensorFlow GPU check
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        
    return device