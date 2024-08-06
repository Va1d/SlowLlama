# Run Llama-3.1-405B on (almost) any PC.
## no advanced skills required.

Steps:

1. Follow the instructions on the GitHub LLama 3.1 official [page](https://github.com/meta-llama/llama3) to download the code and checkpoints
2. when downloaded you'll find the "api" folder that comes along with model checkpoints:
 

![like this](/media//api_folder.png)

3. Download [slowLlama.py](/models/llama3_1/api/slowLlama.py) and place it into the api folder.
4. Run slowLlama.py
5. Wait for a couple of hours the first time while it does some housekeeping, 

	after that, it will be ~10-30min per token generated depending on how much juice your PC has.
6. Brag about running the biggest open-source LLM, 

	uncensored and unconstraint, right on your laptop

## Minimal requirements: 
- Python >= 3.9 
- some NVidia GPU with CUDA and Pytorch installed
- 32Gb memory
- 1Tb of the free space, those checkpoints are Yuge!
 

