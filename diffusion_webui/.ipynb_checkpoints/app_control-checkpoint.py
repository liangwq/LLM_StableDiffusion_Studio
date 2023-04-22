#!/usr/bin/env python

from __future__ import annotations

import os

import gradio as gr
import torch

from app_canny import create_demo as create_demo_canny
from app_depth import create_demo as create_demo_depth
from app_ip2p import create_demo as create_demo_ip2p
from app_lineart import create_demo as create_demo_lineart
from app_mlsd import create_demo as create_demo_mlsd
from app_normal import create_demo as create_demo_normal
from app_openpose import create_demo as create_demo_openpose
from app_scribble import create_demo as create_demo_scribble
from app_scribble_interactive import \
    create_demo as create_demo_scribble_interactive
from app_segmentation import create_demo as create_demo_segmentation
from app_shuffle import create_demo as create_demo_shuffle
from app_softedge import create_demo as create_demo_softedge
from model import Model

DESCRIPTION = '# ControlNet v1.1'

SPACE_ID = os.getenv('SPACE_ID')
ALLOW_CHANGING_BASE_MODEL = SPACE_ID != 'hysts/ControlNet-v1-1'

if SPACE_ID is not None:
    DESCRIPTION += f'\n<p>For faster inference without waiting in queue, you may duplicate the space and upgrade to GPU in settings. <a href="https://huggingface.co/spaces/{SPACE_ID}?duplicate=true"><img style="display: inline; margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3gLdBN6" alt="Duplicate Space" /></a></p>'

if torch.cuda.is_available():
    DESCRIPTION += '\n<p>Running on GPU 🔥</p>'
else:
    DESCRIPTION += '\n<p>Running on CPU 🥶 This demo does not work on CPU.'

MAX_NUM_IMAGES = int(os.getenv('MAX_NUM_IMAGES', '3'))
DEFAULT_NUM_IMAGES = min(MAX_NUM_IMAGES,
                         int(os.getenv('DEFAULT_NUM_IMAGES', '1')))

DEFAULT_MODEL_ID = os.getenv('DEFAULT_MODEL_ID',
                             'runwayml/stable-diffusion-v1-5')
model = Model(base_model_id=DEFAULT_MODEL_ID, task_name='Canny')

with gr.Blocks(css='style.css') as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Tabs():
        with gr.TabItem('Canny'):
            create_demo_canny(model.process_canny,
                              max_images=MAX_NUM_IMAGES,
                              default_num_images=DEFAULT_NUM_IMAGES)
        with gr.TabItem('MLSD'):
            create_demo_mlsd(model.process_mlsd,
                             max_images=MAX_NUM_IMAGES,
                             default_num_images=DEFAULT_NUM_IMAGES)
        with gr.TabItem('Scribble'):
            create_demo_scribble(model.process_scribble,
                                 max_images=MAX_NUM_IMAGES,
                                 default_num_images=DEFAULT_NUM_IMAGES)
        with gr.TabItem('Scribble Interactive'):
            create_demo_scribble_interactive(
                model.process_scribble_interactive,
                max_images=MAX_NUM_IMAGES,
                default_num_images=DEFAULT_NUM_IMAGES)
        with gr.TabItem('SoftEdge'):
            create_demo_softedge(model.process_softedge,
                                 max_images=MAX_NUM_IMAGES,
                                 default_num_images=DEFAULT_NUM_IMAGES)
        with gr.TabItem('OpenPose'):
            create_demo_openpose(model.process_openpose,
                                 max_images=MAX_NUM_IMAGES,
                                 default_num_images=DEFAULT_NUM_IMAGES)
        with gr.TabItem('Segmentation'):
            create_demo_segmentation(model.process_segmentation,
                                     max_images=MAX_NUM_IMAGES,
                                     default_num_images=DEFAULT_NUM_IMAGES)
        with gr.TabItem('Depth'):
            create_demo_depth(model.process_depth,
                              max_images=MAX_NUM_IMAGES,
                              default_num_images=DEFAULT_NUM_IMAGES)
        with gr.TabItem('Normal map'):
            create_demo_normal(model.process_normal,
                               max_images=MAX_NUM_IMAGES,
                               default_num_images=DEFAULT_NUM_IMAGES)
        with gr.TabItem('Lineart'):
            create_demo_lineart(model.process_lineart,
                                max_images=MAX_NUM_IMAGES,
                                default_num_images=DEFAULT_NUM_IMAGES)
        with gr.TabItem('Content Shuffle'):
            create_demo_shuffle(model.process_shuffle,
                                max_images=MAX_NUM_IMAGES,
                                default_num_images=DEFAULT_NUM_IMAGES)
        with gr.TabItem('Instruct Pix2Pix'):
            create_demo_ip2p(model.process_ip2p,
                             max_images=MAX_NUM_IMAGES,
                             default_num_images=DEFAULT_NUM_IMAGES)

    with gr.Accordion(label='Base model', open=False):
        with gr.Row():
            with gr.Column():
                current_base_model = gr.Text(label='Current base model')
            with gr.Column(scale=0.3):
                check_base_model_button = gr.Button('Check current base model')
        with gr.Row():
            with gr.Column():
                new_base_model_id = gr.Text(
                    label='New base model',
                    max_lines=1,
                    placeholder='runwayml/stable-diffusion-v1-5',
                    info=
                    'The base model must be compatible with Stable Diffusion v1.5.',
                    interactive=ALLOW_CHANGING_BASE_MODEL)
            with gr.Column(scale=0.3):
                change_base_model_button = gr.Button(
                    'Change base model', interactive=ALLOW_CHANGING_BASE_MODEL)
        if not ALLOW_CHANGING_BASE_MODEL:
            gr.Markdown(
                '''The base model is not allowed to be changed in this Space so as not to slow down the demo, but it can be changed if you duplicate the Space. <a href="https://huggingface.co/spaces/{SPACE_ID}?duplicate=true"><img style="display: inline; margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3gLdBN6" alt="Duplicate Space" /></a>'''
            )

    check_base_model_button.click(fn=lambda: model.base_model_id,
                                  outputs=current_base_model,
                                  queue=False)
    new_base_model_id.submit(fn=model.set_base_model,
                             inputs=new_base_model_id,
                             outputs=current_base_model)
    change_base_model_button.click(fn=model.set_base_model,
                                   inputs=new_base_model_id,
                                   outputs=current_base_model)

demo.queue(api_open=False, max_size=10).launch(server_port=6006)
