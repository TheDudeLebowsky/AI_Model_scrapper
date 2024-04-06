pip install transformers
pip3 install torch torchvision torchaudio #CPU compute only
pip install huggingface_hub
pip install huggingface_hub[inference]
pip install huggingface_hub["cli"]
python -m venv myenv #to create a virtual environment
myenv\Scripts\activate.bat
CLI :
huggingface-cli login --token $HUGGINGFACE_TOKEN #to login
to delete cache: huggingface-cli delete-cache #run command
