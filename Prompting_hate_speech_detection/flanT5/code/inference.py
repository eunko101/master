import json
import os
import pandas as pd
import re
import time
import timeit
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def model_name_to_model_path(model_name):
    """
    Transforms names into their corresponding paths.
    Examples :
    'mistralai/Mistral-7B-Instruct-v0.2'->'/datasets/huggingface_hub/
    models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/
    27dcfa74d334bc871f3234de431e71c6eeba5dd6'
    '01-ai/Yi-6B' -> '/datasets/huggingface_hub/models--01-ai--Yi-6B/
    snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6'
    """
    model_name = model_name.replace("/", "--")
    model_path = "/datasets/huggingface_hub/models--"
    +model_name
    +"/snapshots/"

    dirs = os.listdir(model_path)
    if len(dirs) == 1:
        return os.path.join(model_path, dirs[0])
    else:
        raise ValueError(f"There is more than one file in the {model_path}")


def get_dict_datasets():

    dataset_map = {
        "SX": {  # sexism data of full train set
            "filepath": "../../../data/sexism_dataset_3_"
            + "classes_without_duplicates_train.csv",
            "sep": "\t",
            "usecols": ["label", "toxicity", "sentence"],
        },
        "SXall": {  # sexim whole dataset without duplicates
            "filepath": "../../data/sexism_dataset_3_"
            + "classes_without_duplicates_2024_08_14.csv",
            "sep": ",",
            "usecols": ["label", "toxicity", "sentence"],
        },
        "SXM": {  # sexism data of 50 most toxic ex
            "filepath": "../../data/output_most_toxic_" + "2024_06_11_53_sentences.csv",
            "sep": ",",
            "usecols": ["label", "toxicity", "sentence"],
        },
        "SXL": {  # sexism data of 50 least toxic ex
            "filepath": "../../data/output_least_toxic_"
            + "2024_06_11_53_sentences.csv",
            "sep": ",",
            "usecols": ["label", "toxicity", "sentence"],
        },
        "CN": {  # conan(islamophbic) data
            "filepath": "../../data/conan_train.csv",
            "sep": "\t",
            "usecols": None,
        },
    }
    return dataset_map


def load_dataset(dataset_name, head=None):
    dataset_map = get_dict_datasets()
    config = dataset_map[dataset_name]
    data = pd.read_csv(config["filepath"], sep=config["sep"], usecols=config["usecols"])
    if dataset_name != "CN":
        data.rename(columns={"label": "gold", "sentence": "text"}, inplace=True)
    # data = data.head(100)
    return data


def charge_example(prompt_template, texts):
    prompts_charged = [prompt_template.format(sent=text) for text in texts]
    return prompts_charged


def turn_into_batch(prompts, batch_size):
    batch_idx = 0
    for i in range(0, len(prompts), batch_size):
        yield prompts[i : i + batch_size]
        batch_idx += 1
        print(f"Processing {batch_idx}th batch")


def inference(batch_of_prompts, tokenizer, model, generation_args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = tokenizer(batch_of_prompts, return_tensors="pt", padding=True).to(device)

    outputs = model.generate(**inputs, **generation_args)
    decoded_outputs = [
        tokenizer.decode(output, skip_special_tokens=True) for output in outputs
    ]
    return decoded_outputs


def store_result(out, prompting_type, directory, data, dataset_name):
    timestr = time.strftime("%Y_%m_%d")
    res_filename = (
        "./output/"
        # + directory
        + f"/dataset_{dataset_name}_"
        + f"{len(out)}_prompt_{prompting_type}_"
        + f"model_{directory}_{timestr}.csv"
    )
    res = pd.concat([data, pd.DataFrame(out, columns=["pred"])], axis=1)
    res.to_csv(res_filename, index=False)
    print(res_filename)
    return res


def get_model(model_name):

    generation_args = {
        "max_length": 10000,
        "temperature": 0.01,
        "top_p": 0.9,
        "top_k": 1,
    }
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return generation_args, tokenizer, model


def main():
    """Main function to test a prompt type chosen."""

    print("run main")
    model_name = "google/flan-t5-large"
    dataset_name = input(
        "Enter the name of the dataset. \n"
        + "How do I do that? \n"
        + "type: \n"
        + "CN for Conan, \n"
        + "SX for Sexism dataset (only half), \n"
        + "SXall for Sexism dataset, \n"
        + "SXM for the top 50 most toxic in the sexims dataset, \n"
        + "SXL for the top 50 least toxic in the sexism dataset. \n"
        + "So give me an answer:"
    )

    list_prompting_type = [
        "P1EN_binary_v1",
        "P2EN_binary_v1",
        "P2EN_binary_v2",
        "P3EN_binary_v2",
        "P1FR_binary_v1",
        "P2FR_binary_v1",
        "P2FR_binary_v2",
        "P3FR_binary_v2",
    ]
    start = timeit.default_timer()
    for model_name in [
        "google/flan-t5-large",
        "google/flan-t5-xl",
        "google/flan-t5-xxl",
    ]:
        print("Model name: ", model_name)
        for prompting_type in list_prompting_type:
            print("Prompt type: ", prompting_type)
            data = load_dataset(dataset_name)
            directory = model_name.split("/")[1]
            generation_args, tokenizer, model = get_model(model_name)

            with open("../../data/prompts.json", "r") as f:
                prompt_templates = json.load(f, strict=False)

            prompts = charge_example(prompt_templates[prompting_type], data["text"])
            out = []
            for batch in turn_into_batch(prompts, batch_size=int(len(prompts) / 2000)):
                out.extend(inference(batch, tokenizer, model, generation_args))

            store_result(out, prompting_type, directory, data, dataset_name)

    stop = timeit.default_timer()
    print("Time: ", stop - start)


if __name__ == "__main__":
    main()
