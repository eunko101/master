# Scripts
## `inference.py`
This script generates prompts and runs inference on the model FlanT5.

### You can choose:
- `model_name`: "google/flan-t5-large" or "google/flan-t5-XL" depending on the size.
- `dataset_name`: Different types of sexism and Conan datasets mapped with their IDs:
  ```python
  dataset_map = {"SX": "sexism full", "CN": "conan", "SXM": "sexism most", "SXL": "sexism least"}

### Main functions
`read_prompt_template` : Load a text file containing the prompt template.
`charge_example` : Charge sentences to prompt template. 
`inference` : Run inference on batches. 
`store_result` : Store model's answer in csv file   

I ran this process looping over different styles of prompts where you can find at the end on bloc of code ## Run Test. 

## eval.py
This script evaluate ouputs of the model. 

### Main functions
`evaluate` : Gives confustion matrix and metrics including accuracy, precision, recall, `1, macro f1 and weighted f1. 
At the end it extract a text of brief file that summarizes the evaluation.

# Output
Tested model size are L and XL of flanT5 and for each directory you can find
- csv file named with dataset_name + number of examples + prompting type
    - dataset_name 
        - "CN": "conan" 
        - "SXM": "sexism most"
        - "SX": "sexism full"
        - "SXL": "sexism least"
    - prompting type = prompt_id + language of prompt
        - prompt_id
            - P1 : zero shot strategy
            - P3 : 4 shot strategy 
        - language of prompt
            - EN : English prompt
            - FR : French prompt
- brief.txt
