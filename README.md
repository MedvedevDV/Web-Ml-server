# A simple local web server on FastAPI.
Server can load a pre-prepared '.csv' files and train 3 models - 'Logistic Regression', 'Random Forest' and 'Decision Tree' on this data

# The following methods are implemented on the server:
1.  **fit(model_name)** - train the model and save to disk by specified name
2.  **predict(model_name)** - predict using a trained and loaded model by name
3.  **load(model_name)** - load the trained model by name to the inference mode
4.  **unload(model_name)** - upload the loaded model by name
5.  **remove(model_name)** - delete a trained model from disk by name
6.  **remove_all()** - delete all trained models from disk

Each training of the model is started in a separate process and consumes this process until it's completion. One process always remains for the server, and trained models are loaded and run on the inference.
