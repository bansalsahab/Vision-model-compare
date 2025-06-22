import zipfile
import os
import io
from PIL import Image, ImageSequence

def convert_tif_to_png_from_zip(zip_path, output_dir="converted_pngs"):
    os.makedirs(output_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file_name in zip_ref.namelist():
            if file_name.lower().endswith(('.tif', '.tiff')):
                try:
                    file_data = zip_ref.read(file_name)
                    base_name = os.path.splitext(os.path.basename(file_name))[0]
                    image_bytes = io.BytesIO(file_data)

                    with Image.open(image_bytes) as img:
                        for i, page in enumerate(ImageSequence.Iterator(img), start=1):
                            output_file = os.path.join(output_dir, f"{base_name}_{i}.png")
                            page.convert("RGB").save(output_file, format="PNG")
                            print(f"✅ Saved: {output_file}")
                except Exception as e:
                    print(f"❌ Error processing {file_name}: {e}")
            else:
                print(f"⏩ Skipped non-TIFF file: {file_name}")

    print(f"\n🎉 All valid TIFFs converted to PNG in: {output_dir}")

if __name__ == "__main__":
    convert_tif_to_png_from_zip("Archive2.zip")
