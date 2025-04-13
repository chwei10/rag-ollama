from multimodal_vector import add_image_document
import sys
import os

# Check if an image path was provided
if len(sys.argv) < 2:
    print("Usage: python test_add_image.py <path_to_image> [caption]")
    sys.exit(1)

image_path = sys.argv[1]
caption = sys.argv[2] if len(sys.argv) > 2 else "Test image"

if not os.path.exists(image_path):
    print(f"Error: Image file not found at {image_path}")
    sys.exit(1)

try:
    print(f"Adding image: {image_path}")
    print(f"Caption: {caption}")
    add_image_document(image_path, caption)
    print("✅ Successfully added image to vector store")
except Exception as e:
    print(f"❌ Error adding image: {e}")
    import traceback
    traceback.print_exc()