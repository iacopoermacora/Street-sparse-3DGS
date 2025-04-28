import subprocess
import sys
import argparse

# Convert COLMAP .bin files to ply
def convert_to_ply(input_path, output_path):
    try:
        subprocess.run(
            [
                "colmap", "model_converter",
                "--input_path", input_path,
                "--output_path", output_path,
                "--output_type", "PLY"
            ],
            check=True
        )
        print(f"Successfully converted model from TXT to BIN. Output saved in: {output_path}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"COLMAP model_converter failed: {e}")
    except FileNotFoundError:
        raise RuntimeError("COLMAP executable not found. Make sure COLMAP is installed and added to your PATH.")
    
# Define input and output paths
if __name__ == "__main__":
    # Accept input and output paths as command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help="Path to input COLMAP model file")
    parser.add_argument("--output_path", help="Path to output PLY file")
    args = parser.parse_args()
    convert_to_ply(args.input_path, args.output_path)