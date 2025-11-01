#!/usr/bin/env python3

import argparse
import concurrent.futures
import json
import logging
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import PIL.Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SizeMode(Enum):
    HEIGHT_WIDTH = "height_width"
    SMALL_LARGE = "small_large"

@dataclass
class ImageDimensions:
    width: int
    height: int

    @property
    def small_side(self) -> int:
        return min(self.width, self.height)

    @property
    def large_side(self) -> int:
        return max(self.width, self.height)

    @property
    def pixel_count(self) -> int:
        return self.width * self.height

    def exceeds_target(self, args: 'Arguments', original_size: int) -> bool:
        """Check if image exceeds target dimensions or size"""
        if args.size_mode == SizeMode.HEIGHT_WIDTH:
            if (self.width > args.max_width or 
                self.height > args.max_height):
                return True
        else:  # SMALL_LARGE mode
            if (self.small_side > args.max_small_side or 
                self.large_side > args.max_large_side):
                return True

        if self.pixel_count > args.max_pixels:
            return True

        size_threshold = original_size / (1 + args.size_tolerance)
        return os.path.getsize(original_size) > size_threshold

@dataclass
class Arguments:
    input_path: Path
    recursive: bool = False
    max_width: int = 1920
    max_height: int = 1080
    max_small_side: int = 1080
    max_large_side: int = 1920
    max_pixels: int = 2073600  # 1920x1080
    size_tolerance: float = 0.1  # 10%
    size_mode: SizeMode = SizeMode.HEIGHT_WIDTH
    backup_ext: Optional[str] = None
    parallel_jobs: int = max(1, multiprocessing.cpu_count() - 1)
    cwebp_args: Dict[str, str] = None

    @classmethod
    def from_argparse(cls, args: argparse.Namespace) -> 'Arguments':
        """Create Arguments instance from argparse namespace"""
        cwebp_args = {
            "mt": None,
            "m": "6",
            "sharp_yuv": None,
            "q": "75",
            "jpeg_like": None,
        }
        
        # Add any additional cwebp arguments from command line
        if hasattr(args, 'cwebp_args'):
            cwebp_args.update(args.cwebp_args)

        return cls(
            input_path=Path(args.input_path),
            recursive=args.recursive,
            max_width=args.max_width,
            max_height=args.max_height,
            max_small_side=args.max_small_side,
            max_large_side=args.max_large_side,
            max_pixels=args.max_pixels,
            size_tolerance=args.size_tolerance,
            size_mode=SizeMode(args.size_mode),
            backup_ext=args.backup_ext,
            parallel_jobs=args.parallel_jobs,
            cwebp_args=cwebp_args
        )

def get_image_dimensions(image_path: Path) -> ImageDimensions:
    """Get image dimensions using PIL"""
    with PIL.Image.open(image_path) as img:
        return ImageDimensions(width=img.width, height=img.height)

def build_cwebp_command(input_path: Path, output_path: Path, args: Arguments) -> List[str]:
    """Build cwebp command with appropriate arguments"""
    cmd = ["cwebp"]
    
    for arg, value in args.cwebp_args.items():
        cmd.append(f"-{arg}")
        if value is not None:
            cmd.append(str(value))
    
    cmd.extend([str(input_path), "-o", str(output_path)])
    return cmd

def process_image(image_path: Path, work_dir: Path, args: Arguments) -> bool:
    """Process a single image, returns True if image was converted"""
    try:
        dims = get_image_dimensions(image_path)
        original_size = os.path.getsize(image_path)
        
        # Check if conversion is needed
        needs_conversion = (
            dims.exceeds_target(args, original_size) or
            image_path.suffix.lower() in {'.png', '.bmp'} or
            original_size > (1024 * 1024)  # 1MB
        )
        
        if not needs_conversion:
            return False
            
        # Convert image
        output_path = work_dir / f"{image_path.stem}.webp"
        cmd = build_cwebp_command(image_path, output_path, args)
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Failed to convert {image_path}: {result.stderr}")
            return False
            
        # Check if new file is worth keeping
        new_size = os.path.getsize(output_path)
        if new_size > original_size * (1 + args.size_tolerance):
            output_path.unlink()
            return False
            
        # Replace original with converted file
        image_path.unlink()
        output_path.rename(image_path)
        return True
        
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")
        return False

def process_cbz(cbz_path: Path, args: Arguments) -> bool:
    """Process a single CBZ file"""
    try:
        # Create temporary working directory
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            
            # Extract CBZ
            with zipfile.ZipFile(cbz_path, 'r') as zip_ref:
                zip_ref.extractall(work_dir)
            
            # Process images in parallel
            changed = False
            image_paths = []
            for ext in ('*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp'):
                image_paths.extend(work_dir.rglob(ext))
            
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.parallel_jobs
            ) as executor:
                futures = {
                    executor.submit(process_image, img_path, work_dir, args): img_path 
                    for img_path in image_paths
                }
                for future in concurrent.futures.as_completed(futures):
                    if future.result():
                        changed = True
            
            if not changed:
                return False
                
            # Backup original if requested
            if args.backup_ext:
                backup_path = cbz_path.with_suffix(f".{args.backup_ext}")
                shutil.move(cbz_path, backup_path)
            
            # Create new CBZ
            with zipfile.ZipFile(cbz_path, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
                for file_path in work_dir.rglob('*.*'):
                    if file_path.is_file():
                        rel_path = file_path.relative_to(work_dir)
                        zip_ref.write(file_path, rel_path)
            
            return True
            
    except Exception as e:
        logger.error(f"Error processing {cbz_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Optimize CBZ files using WebP compression",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "-i", "--input-path",
        required=True,
        help="Input directory containing CBZ files or a single CBZ file"
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Recursively process subdirectories"
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=1920,
        help="Maximum image width"
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=1080,
        help="Maximum image height"
    )
    parser.add_argument(
        "--max-small-side",
        type=int,
        default=1080,
        help="Maximum length of smaller image side"
    )
    parser.add_argument(
        "--max-large-side",
        type=int,
        default=1920,
        help="Maximum length of larger image side"
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=2073600,
        help="Maximum number of pixels (width * height)"
    )
    parser.add_argument(
        "--size-tolerance",
        type=float,
        default=0.1,
        help="Acceptable size increase factor (e.g., 0.1 = 10%)"
    )
    parser.add_argument(
        "--size-mode",
        type=str,
        choices=[m.value for m in SizeMode],
        default=SizeMode.HEIGHT_WIDTH.value,
        help="How to interpret size limits"
    )
    parser.add_argument(
        "--backup-ext",
        help="Extension for backup files (skip backup if not provided)"
    )
    parser.add_argument(
        "-j", "--parallel-jobs",
        type=int,
        default=max(1, multiprocessing.cpu_count() - 1),
        help="Number of parallel jobs"
    )
    
    # Add all cwebp parameters
    cwebp_group = parser.add_argument_group('cwebp options')
    cwebp_params = [
        ('mt', 'Enable multithreading'),
        ('m', 'Compression method (0-6)'),
        ('sharp_yuv', 'Use sharp YUV conversion'),
        ('q', 'Quality factor (0-100)'),
        ('alpha_q', 'Alpha quality factor (0-100)'),
        ('f', 'Filter strength (0-100)'),
        ('preset', 'Preset setting (default|photo|picture|drawing|icon|text)'),
        ('jpeg_like', 'Optimize for JPEG-like quality'),
        ('target_size', 'Target size in bytes'),
        ('target_PSNR', 'Target PSNR'),
        ('segments', 'Number of segments (1-4)'),
        ('sns', 'Spatial noise shaping (0-100)'),
        ('filter_strength', 'Filter strength (0-100)'),
        ('filter_sharpness', 'Filter sharpness (0-7)'),
        ('filter_type', 'Filter type (0-1)'),
        ('autofilter', 'Auto-adjust filter strength'),
        ('alpha_filter', 'Alpha plane filter'),
        ('alpha_method', 'Alpha compression method'),
        ('exact', 'Preserve RGB values'),
        ('blend_alpha', 'Blend alpha channel'),
        ('noalpha', 'Discard alpha channel'),
        ('lossless', 'Enable lossless compression'),
        ('near_lossless', 'Near-lossless compression'),
        ('hint', 'Image hint (photo,picture,graph)'),
        ('metadata', 'Metadata to copy'),
    ]
    
    for param, help_text in cwebp_params:
        cwebp_group.add_argument(
            f'--{param}',
            help=help_text
        )

    args = parser.parse_args()
    config = Arguments.from_argparse(args)
    
    input_path = Path(args.input_path)
    if input_path.is_file() and input_path.suffix.lower() == '.cbz':
        process_cbz(input_path, config)
    elif input_path.is_dir():
        pattern = '**/*.cbz' if args.recursive else '*.cbz'
        for cbz_path in input_path.glob(pattern):
            process_cbz(cbz_path, config)
    else:
        logger.error(f"Invalid input path: {input_path}")
        sys.exit(1)

if __name__ == "__main__":
    main()