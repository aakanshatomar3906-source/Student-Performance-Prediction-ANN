
import json
import zipfile
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

original_model = MODEL_DIR / "ann_model.keras"
fixed_model = MODEL_DIR / "ann_model_compatible.keras"

if not original_model.exists():
    raise FileNotFoundError(
        f"Model not found: {original_model}"
    )

print("Reading original model...")

# Read the Keras archive and repair its configuration
with zipfile.ZipFile(original_model, "r") as source:
    with zipfile.ZipFile(
        fixed_model, "w", compression=zipfile.ZIP_DEFLATED
    ) as destination:

        for item in source.infolist():
            content = source.read(item.filename)

            if item.filename == "config.json":
                config = json.loads(content)

                def remove_quantization_config(obj):
                    if isinstance(obj, dict):
                        obj.pop("quantization_config", None)

                        for value in obj.values():
                            remove_quantization_config(value)

                    elif isinstance(obj, list):
                        for value in obj:
                            remove_quantization_config(value)

                remove_quantization_config(config)

                content = json.dumps(config).encode("utf-8")

            destination.writestr(item, content)

print("Repaired model configuration!")
print("Saved to:", fixed_model)

# Test the repaired model
import keras

model = keras.models.load_model(
    fixed_model,
    compile=False
)

print("\nSUCCESS! Model loaded.")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)
