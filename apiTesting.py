from flask import Flask, request, jsonify
import tensorflow_hub as hub
 
app = Flask(__name__)
model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
 
@app.route('/embed', methods=['POST'])
def embed():
    data = request.json
    sentences = data['sentences']
    embeddings = model(sentences).numpy().tolist()
    return jsonify({'embeddings': embeddings})
 
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
