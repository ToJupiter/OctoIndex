from interface import create_ui, custom_css
import gradio as gr

if __name__ == "__main__":
    app = create_ui()
    app.launch(
        css=custom_css,
        theme=gr.themes.Base(),
        share=True
    )