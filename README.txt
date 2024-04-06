Okayish code to download Huggingface AI models and dataset from a txt file list.
Need major improvements





These needs to be installed
pip install transformers
pip3 install torch torchvision torchaudio #CPU compute only
pip install huggingface_hub
pip install huggingface_hub[inference]
pip install huggingface_hub["cli"]

CLI :
huggingface-cli login --token $HUGGINGFACE_TOKEN #Login
to delete cache: huggingface-cli delete-cache #run command
