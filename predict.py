import cv2 as cv
import os
import torch
import subprocess
import glob
import tarfile
import numpy as np
from typing import List
from diffusers import DiffusionPipeline, ControlNetModel, AutoPipelineForImage2Image
from latent_consistency_controlnet import LatentConsistencyModelPipeline_controlnet
from cog import BasePredictor, Input, Path
from PIL import Image

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        torch_dtype = torch.float16
        torch_device = "cuda"

        self.txt2img_pipe = DiffusionPipeline.from_pretrained(
            "SimianLuo/LCM_Dreamshaper_v7",
            cache_dir="model_cache",
            safety_checker=None,
            local_files_only=True,
        )

        self.txt2img_pipe.to(torch_device=torch_device, torch_dtype=torch_dtype)

        self.img2img_pipe = AutoPipelineForImage2Image.from_pretrained(
            "SimianLuo/LCM_Dreamshaper_v7",
            cache_dir="model_cache",
            safety_checker=None,
            local_files_only=True,
        )

        self.img2img_pipe.to(torch_device="cuda", torch_dtype=torch.float16)

        controlnet_canny = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_canny",
            cache_dir="model_cache",
            local_files_only=True,
            torch_dtype=torch_dtype,
        ).to(torch_device)

        self.img2img_controlnet_pipe = (
            LatentConsistencyModelPipeline_controlnet.from_pretrained(
                "SimianLuo/LCM_Dreamshaper_v7",
                cache_dir="model_cache",
                safety_checker=None,
                controlnet=controlnet_canny,
                local_files_only=True,
                scheduler=None,
            )
        )

        self.img2img_controlnet_pipe.to(
            torch_device=torch_device, torch_dtype=torch_dtype
        )

    def images_to_video(self, image_folder_path, output_video_path, fps, prefix="out"):
        # Forming the ffmpeg command
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file if it exists
            "-framerate",
            str(fps),  # Set the framerate for the input files
            "-pattern_type",
            "glob",  # Enable pattern matching for filenames
            "-i",
            f"{image_folder_path}/{prefix}-*.png",  # Input files pattern
            "-c:v",
            "libx264",  # Set the codec for video
            "-pix_fmt",
            "yuv420p",  # Set the pixel format
            "-crf",
            "17",  # Set the constant rate factor for quality
            output_video_path,  # Output file
        ]

        # Run the ffmpeg command
        subprocess.run(cmd)

    def zoom_image(self, image: Image.Image, zoom_percentage: float) -> Image.Image:
        """Zooms into the image by a given percentage."""
        width, height = image.size
        new_width = width * (1 + zoom_percentage)
        new_height = height * (1 + zoom_percentage)

        # Resize the image to the new dimensions
        zoomed_image = image.resize((int(new_width), int(new_height)))

        # Crop the image to the original dimensions, focusing on the center
        left = (zoomed_image.width - width) / 2
        top = (zoomed_image.height - height) / 2
        right = (zoomed_image.width + width) / 2
        bottom = (zoomed_image.height + height) / 2

        return zoomed_image.crop((left, top, right, bottom))

    def tar_frames(self, frame_paths, tar_path):
        with tarfile.open(tar_path, "w:gz") as tar:
            for frame in frame_paths:
                tar.add(frame)

    def control_image(self, image, canny_low_threshold, canny_high_threshold):
        image = np.array(image)
        canny = cv.Canny(image, canny_low_threshold, canny_high_threshold)
        return  Image.fromarray(canny)

    def predict(
        self,
        start_prompt: str = Input(
            description="Prompt to start with, if not using an image",
            default="Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
        ),
        end_prompt: str = Input(
            description="Prompt to animate towards",
            default="Self-portrait watercolour, a beautiful cyborg with purple hair, 8k",
        ),
        image: Path = Input(
            description="Starting image if not using a prompt",
            default=None,
        ),
        width: int = Input(
            description="Width of output. Lower if out of memory",
            default=512,
        ),
        height: int = Input(
            description="Height of output. Lower if out of memory",
            default=512,
        ),
        iterations: int = Input(
            description="Number of times to repeat the img2img pipeline",
            default=12,
        ),
        prompt_strength: float = Input(
            description="Prompt strength when using img2img. 1.0 corresponds to full destruction of information in image",
            ge=0.0,
            le=1.0,
            default=0.2,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps. Recommend 1 to 8 steps.",
            ge=1,
            le=50,
            default=8,
        ),
        use_canny_control_net: bool = Input(
            description="Use canny edge detection to guide animation",
            default=True,
        ),
        controlnet_conditioning_scale: float = Input(
            description="Controlnet conditioning scale",
            ge=0.1,
            le=4.0,
            default=2.0,
        ),
        control_guidance_start: float = Input(
            description="Controlnet start",
            ge=0.0,
            le=1.0,
            default=0.0,
        ),
        control_guidance_end: float = Input(
            description="Controlnet end",
            ge=0.0,
            le=1.0,
            default=1.0,
        ),
        canny_low_threshold: float = Input(
            description="Canny low threshold",
            ge=1,
            le=255,
            default=100,
        ),
        canny_high_threshold: float = Input(
            description="Canny high threshold",
            ge=1,
            le=255,
            default=200,
        ),
        zoom_increment: int = Input(
            description="Zoom increment percentage for each frame",
            ge=0,
            le=4,
            default=0,
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=8.0
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        return_frames: bool = Input(
            description="Return a tar file with all the frames alongside the video",
            default=False,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        # Removing all temporary frames
        tmp_frames = glob.glob("/tmp/out-*.png")
        for frame in tmp_frames:
            os.remove(frame)

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")

        print(f"Using seed: {seed}")
        torch.manual_seed(seed)

        common_args = {
            "width": width,
            "height": height,
            "guidance_scale": guidance_scale,
            "num_images_per_prompt": 1,
            "lcm_origin_steps": 50,
            "output_type": "pil",
        }

        controlnet_args = {
            "control_guidance_start": control_guidance_start,
            "control_guidance_end": control_guidance_end,
            "controlnet_conditioning_scale": controlnet_conditioning_scale,
        }

        img2img_args = {
            "num_inference_steps": num_inference_steps,
            "prompt": end_prompt,
            "strength": prompt_strength,
        }

        input_image = None

        if image:
            print("img2img mode")
            input_image = Image.open(image)
        else:
            print("txt2img mode")
            txt2img_args = {
                "prompt": start_prompt,
                "num_inference_steps": 8,  # Always want a good starting image
            }
            result = self.txt2img_pipe(**common_args, **txt2img_args).images
            input_image = result[0]

        img2img_args["image"] = input_image

        if use_canny_control_net:
            control_image = self.control_image(
                input_image, canny_low_threshold, canny_high_threshold
            )
            img2img_args["control_image"] = control_image


        last_image_path = None
        last_control_image_path = None
        frame_paths = []

        # Iteratively applying img2img transformations
        for iteration in range(iterations):
            if last_image_path:
                print(f"img2img iteration {iteration}")
                last_image = Image.open(last_image_path)
                img2img_args["image"] = last_image

                if use_canny_control_net:
                    control_image = self.control_image(
                        last_image, canny_low_threshold, canny_high_threshold
                    )

                    img2img_args["control_image"] = control_image

                zoom_increment_mapping = {4: 0.1, 3: 0.05, 2: 0.025, 1: 0.00125}
                if 1 <= zoom_increment <= 4:
                    zoom_factor = zoom_increment_mapping[zoom_increment]
                    img2img_args["image"] = self.zoom_image(
                        img2img_args["image"], zoom_factor
                    )

            # Execute the model pipeline here
            if use_canny_control_net:
                result = self.img2img_controlnet_pipe(
                    **common_args, **img2img_args, **controlnet_args
                ).images

                last_control_image_path = f"/tmp/control-{iteration:06d}.png"
                control_image.save(last_control_image_path)
                frame_paths.append(last_control_image_path)
            else:
                result = self.img2img_pipe(**common_args, **img2img_args).images

            # Save the resulting image for the next iteration
            last_image_path = f"/tmp/out-{iteration:06d}.png"
            result[0].save(last_image_path)
            frame_paths.append(last_image_path)

        # Creating an mp4 video from the images
        video_path = "/tmp/output_video.mp4"
        self.images_to_video("/tmp", video_path, 12)
        paths = [Path(video_path)]

        if use_canny_control_net:
            control_video_path = "/tmp/control_video.mp4"
            self.images_to_video("/tmp", control_video_path, 12, prefix="control")
            paths.append(Path(control_video_path))

        # Tar and return all the frames if return_frames is True
        if return_frames:
            print("Tarring and returning all frames and control images")
            tar_path = "/tmp/frames.tar.gz"
            self.tar_frames(frame_paths, tar_path)
            paths.append(Path(tar_path))

        return paths
