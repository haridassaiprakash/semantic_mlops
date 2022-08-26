from azureml.core import Workspace
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core.model import Model
from threading import Thread
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.exceptions import ComputeTargetException
from azureml.core.model import InferenceConfig
from azureml.core.environment import Environment
from azureml.core.webservice import AciWebservice
from azureml.core.runconfig import DockerConfiguration
from sentence_transformers import SentenceTransformer, util,CrossEncoder
from azureml.core import Experiment,get_run


ia = InteractiveLoginAuthentication(tenant_id='419ea850-427d-4819-8bec-fcc773f331e2')
ws = Workspace.from_config(auth=ia)
print(ws)

#docker_config = DockerConfiguration(use_docker=True)

print('creating the environment ...')
#myenv = Environment.from_conda_specification(name='env', file_path='env.yml')
myenv= Environment.get(ws, "myenv", version=None, label=None)

#myenv.docker.base_image = 'mlopscontainerrr.azurecr.io/azureml/azureml_574ee889f652390c72590d3a04435af7:latest'
#myenv.inferencing_stack_version='latest'
inference_config = InferenceConfig(entry_script='score.py', environment=myenv)
print('environment created!')

print('deploying the ACI service ...')
aci_config = AciWebservice.deploy_configuration(
                  cpu_cores=1,
                  memory_gb=1, auth_enabled=False
                  )



# import os

# download_folder = 'downloaded-files'

# from azureml.core import Experiment, Run
# runn=[]
# diabetes_experiment = ws.experiments['diabetes-workspace']
# for logged_run in diabetes_experiment.get_runs():
#     runn.append(logged_run.id)

# #run = get_run(experiment, run_id, rehydrate=True, clean_up=True)
# from azureml.core import Experiment, Run,get_run

# run=get_run(diabetes_experiment, runn[0], rehydrate=True, clean_up=True)

# # Download files in the "outputs" folder
# run.download_files(prefix='outputs', output_directory=download_folder)

# # Verify the files have been downloaded
# for root, directories, filenames in os.walk(download_folder): 
#     for filename in filenames:  
#         print (os.path.join(root,filename))

# bi_encoder = SentenceTransformer('downloaded-files/outputs/biencoder')
# print(bi_encoder)
# cross_encoder = CrossEncoder('downloaded-files/outputs/crossencoder')


#model1 = Model(ws, "diabetes_model.pkl")
#bi_encoder = SentenceTransformer('downloaded-files/outputs/biencoder')
#cross_encoder = CrossEncoder('downloaded-files/outputs/crossencoder')
#model1 = CrossEncoder('outputs/cross-encoder')

aci_service_name='semantic-search-aci'

model1 = Model(ws, "cross_encoder_model.pkl")

aci_service = Model.deploy(ws, aci_service_name, [model1], inference_config, aci_config)

aci_service.wait_for_deployment(show_output=True)
print('ACI service deployed!')
print(aci_service.state)