import gradio as gr
from backend import Database, Ingestor, RAGAgent
import config

custom_css = """
.gradio-container { background-color: #0f0f0f; }
button.primary { background-color: #3b82f6; color: white; }
textarea { background-color: #1a1a1a; color: white; border: 1px solid #333; }
"""

def create_ui():
    db = Database()
    ingestor = Ingestor(db)
    agent = RAGAgent(db)

    def add_files(files):
        if not files: return "No files selected."
        count = ingestor.process_files(files)
        return f"Processed {count} documents."

    def chat_fn(message, history):
        return agent.chat(message)

    def reset_fn():
        agent.clear_history()
        return []

    def clear_db():
        db.clear()
        return "Database cleared."

    with gr.Blocks(css=custom_css, theme=gr.themes.Base()) as demo:
        with gr.Tabs():
            with gr.Tab("Chat"):
                chatbot = gr.Chatbot(height=600)
                msg = gr.Textbox(placeholder="Ask about your documents...")
                clear_chat = gr.Button("Clear Chat History")
                
                gr.ChatInterface(fn=chat_fn, chatbot=chatbot, textbox=msg)
                clear_chat.click(reset_fn, outputs=[chatbot])

            with gr.Tab("Knowledge Base"):
                file_upload = gr.File(file_count="multiple", label="Upload PDF/MD")
                upload_btn = gr.Button("Process Documents", variant="primary")
                status = gr.Textbox(label="Status", interactive=False)
                clear_btn = gr.Button("Clear Database", variant="stop")

                upload_btn.click(add_files, inputs=[file_upload], outputs=[status])
                clear_btn.click(clear_db, outputs=[status])

    return demo