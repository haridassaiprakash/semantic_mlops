from azureml.core import Workspace, Run
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core import Workspace
print('loaded azureml dependencies!')

print('importing libraries ...')
from transformers import AutoModel
from transformers import AutoTokenizer
import numpy as np
import os
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer, util,CrossEncoder
import torch

print('libraries imported!')




ia = InteractiveLoginAuthentication(tenant_id='419ea850-427d-4819-8bec-fcc773f331e2')
ws=Workspace.from_config(auth=ia)

print(ws)

# Get the experiment run context
run = Run.get_context()


print("Loading Data...")
# load the diabetes dataset

df = pd.read_csv('data/text.csv')
print("dataframe prinnted!")

# We need to install those for model cloning.
#sudo apt-get install git-lfs

#!sudo apt-get install git-lfs
#!git lfs install

# lfs install done
import os
os.makedirs('outputs', exist_ok=True)

os.system(f"apt-get install git-lfs")
os.system(f"git lfs install")
os.system(f"git clone https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-cos-v1")
os.system(f"git clone https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2")

print(os.listdir())
print(os.getcwd())

bi_encoder = AutoModel.from_pretrained('multi-qa-mpnet-base-cos-v1')
bi_encoder.save_pretrained("outputs/biencoder")

bi_encoder_tok = AutoTokenizer.from_pretrained("multi-qa-mpnet-base-cos-v1")
bi_encoder_tok.save_pretrained("outputs/biencoder")



biencoder_model = SentenceTransformer('outputs/biencoder')


cross_encoder = AutoModel.from_pretrained('ms-marco-MiniLM-L-6-v2')
cross_encoder.save_pretrained("outputs/crossencoder")

cross_encoder_tok = AutoTokenizer.from_pretrained("ms-marco-MiniLM-L-6-v2")
cross_encoder_tok.save_pretrained("outputs/crossencoder")

cross_encoder_model = CrossEncoder('outputs/crossencoder')

#bi_encoder = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
#cross_encoder = CrossEncoder('ms-marco-MiniLM-L-6-v2')

docs=[]
for x in df["text"]:
    docs.append(x)


corpus_embeddings = biencoder_model.encode(docs, convert_to_tensor=True, show_progress_bar=True)

run_id = run.id
run.log("run_id",run_id)



#os.makedirs('embeddings', exist_ok=True)

print("saving embeddings")

torch.save(corpus_embeddings,'outputs/embeddings')
print("embeddings saveed!")
embed2=torch.load('outputs/embeddings')
print(type(embed2))
# joblib.dump(value=corpus_embeddings, filename='embeddings/corpus_embeddings')
#joblib.dump(value=model, filename='outputs/diabetes_model.pkl')


joblib.dump(value=biencoder_model, filename='outputs/biencoder_model.pkl')
joblib.dump(value=cross_encoder_model, filename='outputs/cross_encoder_model.pkl')

# joblib.dump(value=cross_encoder, filename='outputs/cross_encoder.pkl')

run.complete()