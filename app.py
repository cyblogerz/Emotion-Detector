from fastbook import load_learner
import gradio as gr
from fastai.vision.all import * 
# Load the model
learn = load_learner('export.pkl')

labels = learn.dls.vocab

def predict(img):
    img=PILImage.create(img)
    pred,pred_idx,probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}


# Define the interface
gr.Interface(fn=predict, inputs=gr.inputs.Image(shape=(512, 512)), outputs=gr.outputs.Label(num_top_classes=4)).launch(share=True)
title= "Emotion Classifier"
examples = ["examples/ex-1angry.jpg","examples/sad.jpg","examples/happy.jpg"]