import pickle
import json
import numpy
from azureml.core.model import Model
from sentence_transformers import SentenceTransformer, util,CrossEncoder
import tensor
import joblib

import os


def init():
    global model

    # load the model from file into a global object
    model_path = Model.get_model_path(model_name="cross_encoder_model.pkl")
    model = joblib.load(model_path)
    model_path1 = Model.get_model_path(model_name="biencoder_model.pkl")
    model1 = joblib.load(model_path)

    corpus_embeddings = torch.load('outputs/embeddings')
# download_folder = 'downloaded-files'

# # Download files in the "outputs" folder
# run.download_files(prefix='outputs', output_directory=download_folder)

# # Verify the files have been downloaded
# for root, directories, filenames in os.walk(download_folder): 
#     for filename in filenames:  
#         print (os.path.join(root,filename))

#bi_encoder = SentenceTransformer('downloaded-files/outputs/biencoder')
#cross_encoder = CrossEncoder('downloaded-files/outputs/crossencoder')
    

def run(raw_data):
    # Get the input data as a numpy array
    data = json.loads(raw_data)['data']
    print(data)
    print(type(data))
    #corpus_embeddings = torch.load('outputs/embeddings')
    #print(model)
    # Get a prediction from the model
    
    top_k = 3
    question_embedding = model1.encode(data, convert_to_tensor=True)
    question_embedding = question_embedding
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k)
    hits = hits[0]  # Get the hits for the first query

    ##### Re-Ranking #####
    # Now, score all retrieved passages with the cross_encoder
    cross_inp = [[query, docs[hit['corpus_id']]] for hit in hits]
    cross_scores = cross_encoder.predict(cross_inp)

    # Sort results by the cross-encoder scores
    for idx in range(len(cross_scores)):
        hits[idx]['cross-score'] = cross_scores[idx]

    # Output of top-5 hits from re-ranker
    print("\n-------------------------\n")
    print("Top-3 Cross-Encoder Re-ranker hits")
    hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
    # for hit in hits[0:3]:
    #     print("\t{:.3f}\t{}".format(hit['cross-score'], docs[hit['corpus_id']].replace("\n", " ")))

    #predictions = model.encode(str(data), convert_to_tensor=True, show_progress_bar=True)
    # Get the corresponding classname for each prediction (0 or 1)
    # Return the predictions as JSON
    #return json.dumps(predictions)
    return hits



# def init():
#     global model

#     # load the model from file into a global object
#     model_path = Model.get_model_path(model_name="diabetes_model.pkl")
#     model = joblib.load(model_path)


# def run(raw_data):
#     try:
#         data = json.loads(raw_data)["data"]
#         data = numpy.array(data)
#         result = model.predict(data)
#         return json.dumps({"result": result.tolist()})
#     except Exception as e:
#         result = str(e)
#         return json.dumps({"error": result})