from flask import Flask, render_template, request

import json
import torch
import clip
import matplotlib.pyplot as plt

from models.load import init_actionclip
from mmaction.utils import register_all_modules

app = Flask(__name__)
register_all_modules(True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = init_actionclip('ViT-B/32-8', device=device)

labels = ['dumping', 'walking']

with torch.no_grad():
    text = clip.tokenize([label for label in labels]).to(device)
    text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)


@app.route('/tensor', methods=['POST'])
# @app.route('/tensor', methods=['GET'])
def tensor():
    if request.method == "POST":
        video_anno = dict(filename='output.mp4', start_index=0)
        video = preprocess(video_anno).unsqueeze(0).to(device)

        with torch.no_grad():
            video_features = model.encode_video(video)

        video_features /= video_features.norm(dim=-1, keepdim=True)
        
        similarity = (100 * video_features @ text_features.T).softmax(dim=-1)
        probs = similarity.cpu().numpy()
        prob = probs.flatten().tolist()
        prob = json.dumps({"dumping" : prob[0], "walking" : prob[1]})

        return prob


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
