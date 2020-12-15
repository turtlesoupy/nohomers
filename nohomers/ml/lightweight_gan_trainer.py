from tqdm.auto import tqdm
import torch
import json
from pathlib import Path
from .cleaner import scores_for_images
from lightweight_gan import Trainer
from lightweight_gan.lightweight_gan import slerp
from uuid import uuid4
from PIL import Image
import tempfile
from torchvision import transforms
import numpy as np
import copy
import random
from dataclasses import dataclass, field
import ffmpeg
from typing import List
import pydash as py_

@dataclass
class GeneratedImage:
    image: Image
    latents: np.ndarray = field(repr=False)


@dataclass
class GeneratedRef:
    name: str
    latents: np.ndarray = field(repr=False)

    def to_dict(self):
        return {
            "image_name": self.name,
            "latent": self.latents.tolist(),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            name=d["image_name"],
            latents=np.array(d["latent"])
        )


@dataclass
class GeneratedTransition:
    dest_index: int 
    dest_name: str
    video_name: str

    def to_dict(self):
        return {
            "dest_index": dst_i,
            "dest_name": dst["image_name"],
            "video_name": str(video_name),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            dest_index=d["dest_index"],
            dest_name=d["image_name"],
            video_name=d["image_name"],

        )


@dataclass
class GeneratedRefWithTransitions:
    name: str
    transitions: List[GeneratedTransition]

    def to_dict(self):
        return {
            "name": self.name,
            "latent": self.latents.tolist(),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            name=d["image_name"],
            latents=np.array(d["latent"])
        )


def get_trainer(
    data='./data',
    results_dir='./results',
    models_dir='./models',
    name='default',
    new=False,
    load_from=-1,
    image_size=256,
    optimizer='adam',
    fmap_max=512,
    transparent=False,
    batch_size=10,
    gradient_accumulate_every=4,
    num_train_steps=150000,
    learning_rate=2e-4,
    save_every=1000,
    evaluate_every=1000,
    generate=False,
    generate_interpolation=False,
    attn_res_layers=[32],
    sle_spatial=False,
    disc_output_size=1,
    antialias=False,
    interpolation_num_steps=100,
    save_frames=False,
    num_image_tiles=8, 
    trunc_psi=0.75,
    aug_prob=None,
    aug_types=['cutout', 'translation'],
    dataset_aug_prob=0.,
    multi_gpus=False,
    calculate_fid_every=None,
    seed=42,
    amp=False
):
    def cast_list(el):
        return el if isinstance(el, list) else [el]

    model_args = dict(
        name=name,
        results_dir=results_dir,
        models_dir=models_dir,
        batch_size=batch_size,
        gradient_accumulate_every=gradient_accumulate_every,
        attn_res_layers=cast_list(attn_res_layers),
        sle_spatial=sle_spatial,
        disc_output_size=disc_output_size,
        antialias=antialias,
        image_size=image_size,
        optimizer=optimizer,
        fmap_max=fmap_max,
        transparent=transparent,
        lr=learning_rate,
        save_every=save_every,
        evaluate_every=evaluate_every,
        trunc_psi=trunc_psi,
        aug_prob=aug_prob,
        aug_types=cast_list(aug_types),
        dataset_aug_prob=dataset_aug_prob,
        calculate_fid_every=calculate_fid_every,
        amp=amp
    )

    ret = Trainer(**model_args)
    ret.load(load_from)
    return ret


@torch.no_grad()
def generate_images(trainer, num=1):
    trainer.GAN.eval()
    latent_dim = trainer.GAN.latent_dim
    image_size = trainer.GAN.image_size
    latents = torch.randn((num, latent_dim)).cuda(trainer.rank)
    generated_images = trainer.generate_truncated(trainer.GAN.GE, latents)
    return list(
        GeneratedImage(
            image=transforms.ToPILImage()(generated_images[i, :, :, :].cpu()),
            latents=latents[i, :],
        )
        for i in range(num)
    )


@torch.no_grad()
def generate_interpolation_frames(trainer, latents_low, latents_high, num_frames, batch_size=50, device="cuda:0"):
    trainer.GAN.eval()
    num_rows = 1

    latent_dim = trainer.GAN.latent_dim
    image_size = trainer.GAN.image_size

    # latents and noise

    #latents_low = torch.randn(num_rows ** 2, latent_dim).cuda(self.rank)
    #latents_high = torch.randn(num_rows ** 2, latent_dim).cuda(self.rank)
    ratios = torch.linspace(0., 1., num_frames) 

    chunks = list(py_.chunk(ratios, size=batch_size))

    latents_low = latents_low.unsqueeze(0)
    latents_high = latents_high.unsqueeze(0)

    ret = []
    for i, chunk_ratios in enumerate(chunks):
        batch_latents = []
        for j, ratio in enumerate(chunk_ratios):
            if (i + j) == 0:
                interp_latents = latents_low
            elif (i + j) == len(ratios) - 1:
                interp_latents = latents_high
            else:
                interp_latents = slerp(ratio, latents_low, latents_high)
            batch_latents.append(interp_latents)
        
        stacked_latents = torch.vstack(batch_latents).to(device)
        generated_images = trainer.generate_truncated(trainer.GAN.GE, stacked_latents)
            
        for b in range(chunk_ratios.size(0)):
            ret.append(transforms.ToPILImage()(generated_images[b, :, :, :].cpu()))
    
    return ret

def frames_to_video(frames, output_path, fps=30, bitrate="1M"):
    with tempfile.TemporaryDirectory() as td:
        for i, frame in enumerate(frames):
            frame.save(Path(td) / f"{i:06d}.jpg")
            
        (
            ffmpeg
            .input(f'{td}/*.jpg', pattern_type='glob', framerate=fps)
            .output(filename=output_path, video_bitrate=bitrate)
            .overwrite_output()
            .run()
        )


def gen_images_and_manifest(trainer, output_base_dir, num=10, batch_size=100, cleaner=None, clean_threshold=None):
    image_output_dir = Path(output_base_dir) / "images"
    image_output_dir.mkdir(exist_ok=True)
    
    image_objects = []
    pbar = tqdm(total=num)
    while len(image_objects) < num:
        images = list(generate_images(trainer, num=batch_size))
        if cleaner and clean_threshold:
            scores = scores_for_images(cleaner, [image.image for image in images])
            images = [
                im for im, score in zip(images, scores)
                if score > clean_threshold
            ]

        for image in images:
            name = f"{uuid4()}.jpg"
            image.image.save(str(image_output_dir / name))
            image_objects.append(GeneratedRef(
                name=name,
                latents=image.latents.cpu().numpy(),
            ))
        pbar.update(len(images))

    pbar.close()
            
    return image_objects[:num]


def gen_interpolation_videos(
    trainer, 
    images: List[GeneratedRef], 
    output_base_dir, 
    per_edge=1, 
    video_duration=2.0, 
    video_fps=30,
    batch_size=100,
) -> List[GeneratedRefWithTransitions]:
    assert len(images) > per_edge
    
    num_frames = int(video_fps * video_duration)
    
    videos_path = Path(output_base_dir) / "videos"
    videos_path.mkdir(exist_ok=True)

    ret = []
    for src_i, src in enumerate(tqdm(images)):
        dest_set = set()
        while len(dest_set) < per_edge:
            i = random.randint(0, len(images) - 1)
            if i != src_i:
                dest_set.add(i)
                
        transitions = []
        
        src_latent = torch.tensor(src.latents).cuda()
        for dst_i in dest_set:
            dst: GeneratedRef = images[dst_i]
            dst_latent = torch.tensor(dst.latents).cuda()
            
            video_name = f"{src.name}_to_{dst.name}.mp4"
            
            # This works in batches
            video_frames = generate_interpolation_frames(
                trainer, 
                latents_low=src_latent,
                latents_high=dst_latent,
                num_frames=num_frames,
                batch_size=batch_size,
            )
            
            frames_to_video(video_frames, output_path=videos_path / video_name, fps=video_fps, bitrate="1M")
            
            transitions.append(
                GeneratedTransition(
                    dest_index=dst_i,
                    dest_name=dst.name,
                    video_name=str(video_name),
                )
            )

        ret.append(GeneratedRefWithTransitions(
            name=src.name,
            transitions=transitions,
        ))
            
    return ret