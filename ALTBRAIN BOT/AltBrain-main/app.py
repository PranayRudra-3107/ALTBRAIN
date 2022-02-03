# Import necessary libraries
from flask import Flask, jsonify, request
from sentence_transformers import SentenceTransformer, util
import pickle
import os
import numpy as np

# Run after server set-up, before receiving requests
if not os.path.isfile("model.pkl"):

        # Download Deep Learning Model
        embedder =  SentenceTransformer('msmarco-distilbert-base-tas-b')

        # Store the downloaded model as pickle file
        with open("model.pkl","wb") as pickle_file:
                pickle.dump(embedder, pickle_file)

else:
        # Load model if already downloaded
        with open("model.pkl","rb") as pickle_file:
                embedder = pickle.load(pickle_file)


# Initialize Flask APP
app = Flask(__name__)

# METHOD for WRITE request
@app.route('/write',methods=['POST'])
def add_record():
        try:
                # return response
                answer = "SUCCESS"
                # print(type(str(request.data)))
                # print(str(request.data))
                # read user input
                # file = request.files['input'].read()
                # print(file)
                # print(str(file))
                input_passage = [str(request.data)]

                # Check if pickle files are present
                if not os.path.isfile("embeddings/passage_embeddings.pkl"):
                        passage_embeddings = embedder.encode(input_passage)

                        with open("embeddings/passage_embeddings.pkl","wb") as pickle_file:
                                pickle.dump(passage_embeddings, pickle_file)

                        with open("passage/passage.pkl","wb") as pickle_file:
                                pickle.dump(input_passage, pickle_file)

                else:
                        with open("embeddings/passage_embeddings.pkl","rb") as pickle_file:
                                passage_embeddings = pickle.load(pickle_file)

                        with open("passage/passage.pkl","rb") as pickle_file:
                                passage = pickle.load(pickle_file)

                        input_passage_embedding = embedder.encode(input_passage)[0]

                        # Concatenate new embeddings
                        new_embeddings = np.vstack((passage_embeddings,input_passage_embedding))
                        #print(len(new_embeddings))
                        #print(passage)

                        passage.append(input_passage[0])
                        #print('working')

                        # Store passage along with its updated embeddings as pickle files
                        with open("embeddings/passage_embeddings.pkl","wb") as pickle_file:
                                pickle.dump(new_embeddings, pickle_file)

                        with open("passage/passage.pkl","wb") as pickle_file:
                                pickle.dump(passage, pickle_file)

        # Handle exception
        except Exception as e:
                print(e)
                answer = "Something wrong"

        return {"result":answer}


# METHOD for READ request
@app.route('/read',methods=['POST'])
def read_record():

        # load passages and corresponding embeddings
        with open("embeddings/passage_embeddings.pkl","rb") as pickle_file:
                passage_embeddings = pickle.load(pickle_file)

        with open("passage/passage.pkl","rb") as pickle_file:
                passage = pickle.load(pickle_file)

        # read user input
        # file = request.files['input'].read()
        query = str(request.data)
        query_embedding = embedder.encode(query)

        # Perform semantic search over text corpus
        # Return top 3 passage matches
        hits = util.semantic_search(query_embedding, passage_embeddings, top_k=3)
        hits = hits[0]
        output = []

        # create response object
        for hit in hits:
                output.append({"sentence":str(passage[hit['corpus_id']]),"score":hit['score']})

        # return output to user
        return {'result':output}


# RUN FLASK APP
if __name__ == '__main__':
    app.run(debug=True)


# Sample Write request
# url = "http://127.0.0.1:5000/write"
# data = {'input': 'London population is the subject being discussed'}

# send_request = requests.post(url,files=data).json()
# print(send_request)


# Sample Read request
# url = "http://127.0.0.1:5000/read"
# data = {'input': 'How many people are there in London?'}

# send_request = requests.post(url,files=data).json()
# print(send_request)
