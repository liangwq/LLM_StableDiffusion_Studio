from __future__ import annotations
import gradio as gr


import os

import torch

from diffusion_webui.controlnet_v1_1.app_canny import create_demo as create_demo_canny
from diffusion_webui.controlnet_v1_1.app_depth import create_demo as create_demo_depth
from diffusion_webui.controlnet_v1_1.app_ip2p import create_demo as create_demo_ip2p
from diffusion_webui.controlnet_v1_1.app_lineart import create_demo as create_demo_lineart
from diffusion_webui.controlnet_v1_1.app_mlsd import create_demo as create_demo_mlsd
from diffusion_webui.controlnet_v1_1.app_normal import create_demo as create_demo_normal
from diffusion_webui.controlnet_v1_1.app_openpose import create_demo as create_demo_openpose
from diffusion_webui.controlnet_v1_1.app_scribble import create_demo as create_demo_scribble
from diffusion_webui.controlnet_v1_1.app_scribble_interactive import \
    create_demo as create_demo_scribble_interactive
from diffusion_webui.controlnet_v1_1.app_segmentation import create_demo as create_demo_segmentation
from diffusion_webui.controlnet_v1_1.app_shuffle import create_demo as create_demo_shuffle
from diffusion_webui.controlnet_v1_1.app_softedge import create_demo as create_demo_softedge
from diffusion_webui.controlnet_v1_1.model import Model

from diffusion_webui.helpers import (
    #CodeformerUpscalerGenerator,
    StableDiffusionControlInpaintNetDepthGenerator,
    StableDiffusionControlNetCannyGenerator,
    StableDiffusionControlNetDepthGenerator,
    StableDiffusionControlNetHEDGenerator,
    StableDiffusionControlNetInpaintCannyGenerator,
    StableDiffusionControlNetInpaintHedGenerator,
    StableDiffusionControlNetInpaintMlsdGenerator,
    StableDiffusionControlNetInpaintPoseGenerator,
    StableDiffusionControlNetInpaintScribbleGenerator,
    StableDiffusionControlNetInpaintSegGenerator,
    StableDiffusionControlNetMLSDGenerator,
    StableDiffusionControlNetPoseGenerator,
    StableDiffusionControlNetScribbleGenerator,
    StableDiffusionControlNetSegGenerator,
    StableDiffusionImage2ImageGenerator,
    StableDiffusionInpaintGenerator,
    StableDiffusionText2ImageGenerator,
)
from llm_chatglm import chatglmControl
#from langchain_ChatGLM import webui

def main():
    DESCRIPTION = '# LLM StableDiffusion Work Studio v0.1'

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
    app = gr.Blocks()
    with app:
        gr.Markdown(DESCRIPTION)
        with gr.Row():
            with gr.Column():
                with gr.Tab("Text2Chat"):
                    chatglmControl.create_demo()
                '''with gr.Tab("Text2ChatLangChain"):
                    webui.create_demo()'''
                with gr.Tab("Text2Img"):
                    StableDiffusionText2ImageGenerator.app()
                with gr.Tab("Img2Img"):
                    StableDiffusionImage2ImageGenerator.app()
                with gr.Tab("Inpaint"):
                    StableDiffusionInpaintGenerator.app()
                with gr.Tab("ControlNet"):
                    with gr.Tab("Canny"):
                        StableDiffusionControlNetCannyGenerator.app()
                    with gr.Tab("Depth"):
                        StableDiffusionControlNetDepthGenerator.app()
                    with gr.Tab("HED"):
                        StableDiffusionControlNetHEDGenerator.app()
                    with gr.Tab("MLSD"):
                        StableDiffusionControlNetMLSDGenerator.app()
                    with gr.Tab("Pose"):
                        StableDiffusionControlNetPoseGenerator.app()
                    with gr.Tab("Scribble"):
                        StableDiffusionControlNetScribbleGenerator.app()
                    with gr.Tab("Seg"):
                        StableDiffusionControlNetSegGenerator.app()
                with gr.Tab("ControlNet Inpaint"):
                    with gr.Tab("Canny"):
                        StableDiffusionControlNetInpaintCannyGenerator.app()
                    with gr.Tab("Depth"):
                        StableDiffusionControlInpaintNetDepthGenerator.app()
                    with gr.Tab("HED"):
                        StableDiffusionControlNetInpaintHedGenerator.app()
                    with gr.Tab("MLSD"):
                        StableDiffusionControlNetInpaintMlsdGenerator.app()
                    with gr.Tab("Pose"):
                        StableDiffusionControlNetInpaintPoseGenerator.app()
                    with gr.Tab("Scribble"):
                        StableDiffusionControlNetInpaintScribbleGenerator.app()
                    with gr.Tab("Seg"):
                        StableDiffusionControlNetInpaintSegGenerator.app()
                '''with gr.Tab("Upscaler"):
                    CodeformerUpscalerGenerator.app()'''
                with gr.Tab('Ctronl v1.1'):
                    with gr.Tab('Canny'):
                        create_demo_canny(model.process_canny,
                                  max_images=MAX_NUM_IMAGES,
                                  default_num_images=DEFAULT_NUM_IMAGES)
                    with gr.Tab('MLSD'):
                        create_demo_mlsd(model.process_mlsd,
                                  max_images=MAX_NUM_IMAGES,
                                  default_num_images=DEFAULT_NUM_IMAGES)
                    with gr.Tab('Scribble'):
                        create_demo_scribble(model.process_scribble,
                                  max_images=MAX_NUM_IMAGES,
                                  default_num_images=DEFAULT_NUM_IMAGES)
                    with gr.Tab('Scribble Interactive'):
                        create_demo_scribble_interactive(
                                  model.process_scribble_interactive,
                                  max_images=MAX_NUM_IMAGES,
                                  default_num_images=DEFAULT_NUM_IMAGES)
                    with gr.Tab('SoftEdge'):
                        create_demo_softedge(model.process_softedge,
                                  max_images=MAX_NUM_IMAGES,
                                  default_num_images=DEFAULT_NUM_IMAGES)
                    with gr.Tab('OpenPose'):
                        create_demo_openpose(model.process_openpose,
                                  max_images=MAX_NUM_IMAGES,
                                  default_num_images=DEFAULT_NUM_IMAGES)
                    with gr.Tab('Segmentation'):
                        create_demo_segmentation(model.process_segmentation,
                                    max_images=MAX_NUM_IMAGES,
                                    default_num_images=DEFAULT_NUM_IMAGES)
                    with gr.Tab('Depth'):
                        create_demo_depth(model.process_depth,
                                    max_images=MAX_NUM_IMAGES,
                                    default_num_images=DEFAULT_NUM_IMAGES)
                    with gr.Tab('Normal map'):
                        create_demo_normal(model.process_normal,
                                    max_images=MAX_NUM_IMAGES,
                                    default_num_images=DEFAULT_NUM_IMAGES)
                    with gr.Tab('Lineart'):
                        create_demo_lineart(model.process_lineart,
                                    max_images=MAX_NUM_IMAGES,
                                    default_num_images=DEFAULT_NUM_IMAGES)
                    with gr.Tab('Content Shuffle'):
                        create_demo_shuffle(model.process_shuffle,
                                    max_images=MAX_NUM_IMAGES,
                                    default_num_images=DEFAULT_NUM_IMAGES)
                    with gr.Tab('Instruct Pix2Pix'):
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
    app.queue(concurrency_count=1)
    app.launch(share=False, inbrowser=True,debug=True, enable_queue=True,server_port=6006)


if __name__ == "__main__":
    main()
    
