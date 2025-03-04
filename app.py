import gradio as gr
from generator import generate_response

def rag_api(query):
    return generate_response(query)

iface = gr.Interface(fn=rag_api, inputs="text", outputs="text")
iface.launch()
