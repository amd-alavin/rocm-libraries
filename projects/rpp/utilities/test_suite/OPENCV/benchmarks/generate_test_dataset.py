#!/usr/bin/env python3
"""
MIT License

Copyright (c) 2019 - 2026 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Generate synthetic test images for OpenCV benchmarking.
Creates test images with various patterns and colors.
"""

import os
import sys
import argparse
from PIL import Image, ImageDraw
import random
import math
import numpy as np

# Default configuration
DEFAULT_OUTPUT_DIR = "input_images_dataset"
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_NUM_IMAGES = 1


def create_output_dir(output_dir):
    """Create output directory if it doesn't exist."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, output_dir)
    os.makedirs(output_path, exist_ok=True)
    return output_path


def generate_gradient_image(index, output_path, width, height):
    """Generate a gradient image."""
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)

    # Random gradient direction
    for y in range(height):
        r = int((y / height) * 255)
        g = int(((height - y) / height) * 255)
        b = int((index * 2) % 255)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    filename = os.path.join(output_path, f"gradient_{index:03d}.jpg")
    img.save(filename, "JPEG", quality=95)
    return filename


def generate_checkerboard_image(index, output_path, width, height):
    """Generate a checkerboard pattern."""
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)

    tile_size = 40 + (index * 2) % 60
    color1 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    color2 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            if ((x // tile_size) + (y // tile_size)) % 2 == 0:
                draw.rectangle([x, y, x + tile_size, y + tile_size], fill=color1)
            else:
                draw.rectangle([x, y, x + tile_size, y + tile_size], fill=color2)

    filename = os.path.join(output_path, f"checkerboard_{index:03d}.png")
    img.save(filename, "PNG")
    return filename


def generate_circle_image(index, output_path, width, height):
    """Generate random circles."""
    bg_color = (random.randint(0, 100), random.randint(0, 100), random.randint(0, 100))
    img = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)

    num_circles = 10 + (index % 20)
    for _ in range(num_circles):
        x = random.randint(0, width)
        y = random.randint(0, height)
        radius = random.randint(20, min(width, height) // 10)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius], fill=color, outline=color
        )

    filename = os.path.join(output_path, f"circles_{index:03d}.jpg")
    img.save(filename, "JPEG", quality=95)
    return filename


def generate_noise_image(index, output_path, width, height):
    """Generate random noise."""
    # Vectorized numpy array creation is 100-1000x faster than pixel loops
    noise_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(noise_array, "RGB")

    filename = os.path.join(output_path, f"noise_{index:03d}.png")
    img.save(filename, "PNG")
    return filename


def generate_stripes_image(index, output_path, width, height):
    """Generate striped pattern."""
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)

    stripe_width = 10 + (index * 3) % 50
    horizontal = index % 2 == 0

    color1 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    color2 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    if horizontal:
        for y in range(0, height, stripe_width * 2):
            draw.rectangle([0, y, width, y + stripe_width], fill=color1)
    else:
        for x in range(0, width, stripe_width * 2):
            draw.rectangle([x, 0, x + stripe_width, height], fill=color2)

    filename = os.path.join(output_path, f"stripes_{index:03d}.jpg")
    img.save(filename, "JPEG", quality=95)
    return filename


def generate_solid_color_image(index, output_path, width, height):
    """Generate solid color image."""
    r = (index * 17) % 256
    g = (index * 31) % 256
    b = (index * 47) % 256

    img = Image.new("RGB", (width, height), color=(r, g, b))

    filename = os.path.join(output_path, f"solid_{index:03d}.jpg")
    img.save(filename, "JPEG", quality=95)
    return filename


def generate_radial_pattern_image(index, output_path, width, height):
    """Generate radial pattern from center."""
    cx, cy = width // 2, height // 2
    max_dist = math.sqrt(cx**2 + cy**2)

    # Vectorized computation using numpy meshgrid
    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)

    # Calculate distance from center for all pixels at once
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    intensity = (dist / max_dist * 255).astype(np.uint8)

    # Calculate RGB channels
    r = (intensity + index * 10) % 256
    g = (255 - intensity) % 256
    b = (intensity * 2) % 256

    # Stack channels to create RGB image
    img_array = np.stack([r, g, b], axis=-1).astype(np.uint8)
    img = Image.fromarray(img_array, "RGB")

    filename = os.path.join(output_path, f"radial_{index:03d}.png")
    img.save(filename, "PNG")
    return filename


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Generate synthetic test images for OpenCV benchmarking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-W", "--width",
        type=int,
        default=DEFAULT_WIDTH,
        help="Image width in pixels"
    )
    parser.add_argument(
        "-H", "--height",
        type=int,
        default=DEFAULT_HEIGHT,
        help="Image height in pixels"
    )
    parser.add_argument(
        "-n", "--num-images",
        type=int,
        default=DEFAULT_NUM_IMAGES,
        help="Number of images to generate"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for generated images"
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.width <= 0 or args.height <= 0:
        print("Error: Width and height must be positive integers", file=sys.stderr)
        sys.exit(1)
    if args.num_images <= 0:
        print("Error: Number of images must be a positive integer", file=sys.stderr)
        sys.exit(1)

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("OpenCV Benchmark Test Dataset Generator")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Resolution: {args.width}x{args.height}")
    print(f"  Number of images: {args.num_images}")
    print(f"  Output directory: {args.output_dir}/")
    print(f"  Random seed: {args.seed}")
    print()

    output_path = create_output_dir(args.output_dir)

    # Distribution of image types
    generators = [
        generate_gradient_image,
        generate_checkerboard_image,
        generate_circle_image,
        generate_noise_image,
        generate_stripes_image,
        generate_solid_color_image,
        generate_radial_pattern_image,
    ]

    for i in range(args.num_images):
        # Cycle through different generators
        generator = generators[i % len(generators)]
        filename = generator(i, output_path, args.width, args.height)

        if args.num_images > 10 and (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{args.num_images} images...")
        elif args.num_images <= 10:
            print(f"Generated: {os.path.basename(filename)}")

    print(f"\n✓ Successfully generated {args.num_images} images!")
    print(f"✓ Location: {output_path}")
    print(f"✓ Resolution: {args.width}x{args.height}")
    print(f"\nDataset is ready for benchmarking!")
    print("\nTo run the benchmark:")
    print(f"  cd {os.path.dirname(os.path.abspath(__file__))}")
    print("  ./build/opencv_vs_rpp_host_hip_benchmarking")


if __name__ == "__main__":
    main()
