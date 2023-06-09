import gradio as gr

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

def main():
    app = gr.Blocks()
    with app:
        with gr.Row():
            with gr.Column():
                with gr.Tab("Text2Chat"):
                    chatglmControl.create_demo()
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

    app.queue(concurrency_count=1)
    app.launch(share=False, inbrowser=True,debug=True, enable_queue=True,server_port=6006)


if __name__ == "__main__":
    main()
