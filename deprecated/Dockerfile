FROM allennlp/allennlp:latest
RUN pip install allennlp-models
RUN pip install fastcoref
RUN python -m spacy download en_core_web_sm
RUN pip install -U huggingface_hub
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('biu-nlp/f-coref', local_dir='/root/.cache/f-coref')"