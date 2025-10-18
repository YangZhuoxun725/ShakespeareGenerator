A simple language model that generates Shakespeare made in python with pytorch.

How to use:
1. Download the files to a known directory
2. Download and open anaconda prompt
3. Navigate to the folder where the files are located
4. Run "conda env create -f environment.yml"
5. Run "conda activate shakespeare-gpt"
6. Open your preferred python IDE and select the shakespeare-gpt environment as the environment/interpreter
7. To train new models open v2.py and edit the variables at the top (batch_size, block_size, max_iters, etc) and run it. This may take some time depending on the number of iterations
8. To generate using the model, open generate.py and change the model name in torch.load('model-5000.pt', weights_only=False) to whatever your model is named and change the max_new_tokens variable to how many character you would like to generate and run the file
