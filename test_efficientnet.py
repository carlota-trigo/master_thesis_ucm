import tensorflow as tf

# Test EfficientNet creation
model = tf.keras.applications.efficientnet.EfficientNetB1(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3)
)

print("Model input shape:", model.input_shape)
print("Number of layers:", len(model.layers))

# Find stem_conv layer
stem_conv_layer = None
for layer in model.layers:
    if 'stem_conv' in layer.name and 'pad' not in layer.name:
        stem_conv_layer = layer
        break

if stem_conv_layer:
    print("Found stem_conv layer:", stem_conv_layer.name)
    weights = stem_conv_layer.get_weights()
    if weights:
        print("Stem conv weights shape:", weights[0].shape)
    else:
        print("No weights found")
else:
    print("Stem conv layer not found")

