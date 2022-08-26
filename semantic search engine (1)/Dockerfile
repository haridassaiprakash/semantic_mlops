FROM python:3.6

RUN apt-get update -y \
    && apt-get install git-lfs
    
RUN apt-get install git-lfs
RUN git lfs install
RUN git clone https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-cos-v1

# Conda Environment
ENV MINICONDA_VERSION py37_4.9.2
ENV PATH /opt/miniconda/bin:$PATH
RUN wget -qO /tmp/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -bf -p /opt/miniconda && \
    conda clean -ay && \
    rm -rf /opt/miniconda/pkgs && \
    rm /tmp/miniconda.sh && \
    find / -type d -name __pycache__ | xargs rm -rf