"""
gradcam.py

Grad-CAM implementation for visualizing model attention on currency images.
Supports multi-class (softmax) models by accepting a class_index parameter,
so the heatmap explains whichever class was actually predicted.
"""

import tensorflow as tf
import numpy as np
import cv2


def make_gradcam_heatmap(img_array, model, base_model_name="mobilenetv2_1.00_224",
                          last_conv_layer_name="out_relu", class_index=0):
    """
    Args:
        img_array: preprocessed image, shape (1, H, W, 3)
        model: trained Keras model (with nested base_model)
        base_model_name: name of the nested backbone layer
        last_conv_layer_name: name of the last conv layer inside the backbone
        class_index: which output class to explain (0=Fake, 1=Not A Note,
            2=Real for the 3-class model; for the old binary model, use 0)
    """
    base_model = model.get_layer(base_model_name)

    conv_model = tf.keras.Model(base_model.input, base_model.get_layer(last_conv_layer_name).output)

    head_input = tf.keras.Input(shape=conv_model.output_shape[1:])
    x = head_input
    found_base = False
    for layer in model.layers:
        if layer.name == base_model_name:
            found_base = True
            continue
        if found_base:
            x = layer(x)
    head_model = tf.keras.Model(head_input, x)

    with tf.GradientTape() as tape:
        conv_output = conv_model(img_array)
        tape.watch(conv_output)
        predictions = head_model(conv_output)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()


def overlay_heatmap(original_img, heatmap, alpha=0.4):
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlayed = heatmap_colored * alpha + original_img * (1 - alpha)
    return overlayed.astype(np.uint8)