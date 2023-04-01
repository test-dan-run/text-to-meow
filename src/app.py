import logging

import gradio as gr
from config import BaseConfig
from predict import inputs, outputs, predict

if __name__ == "__main__":
    logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s")
    config = BaseConfig()

    app = gr.Interface(
        predict,
        inputs=inputs,
        outputs=outputs,
        title="Text-to-Meow",
        description="Ever thought of whether your cat understands your words? It no longer matters! Now you get to speak in their language!",
    )
    
    app.launch(
        server_name="0.0.0.0",
        server_port=config.port,
        enable_queue=True,
        share=True
    )