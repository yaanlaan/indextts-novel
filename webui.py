import gradio as gr

from ui import *


def create_demo():
    with gr.Blocks(title="Indextts-Novel Demo") as demo:
        gr.HTML('''
        <h2><center>Indextts-Novel: Long Text to Speech for Novels and Stories</h2>
<p align="center">
<a href='https://arxiv.org/abs/2506.21619'><img src='https://img.shields.io/badge/ArXiv-2506.21619-red'></a>
</p>
        ''')

        create_main_page(demo)
        create_long_text_page(demo)

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.queue(20)
    demo.launch(server_name=cmd_args.host, server_port=cmd_args.port)