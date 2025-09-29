Prediction model.

Run the gui in the react directory through `npm run dev`.

Run the triton server:
`docker run --gpus=1 --rm -p8003:8000 -p8004:8001 -p8005:8002 -v ${PWD}/../ImageClassifier:/models nvcr.io/nvidia/tritonserver:25.01-py3 tritonserver --model-repository=/models --model-control-mode explicit --load-model pokemon_prediction_model`
Must have docker desktop to run on windows.

Run the server `file_server.py` while in the FileServer directory. This server handles storing files, sending classification requests, and also contains rest endpoints for the front end to communicate to.


