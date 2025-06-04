"""
Question: What is the max bs i can send for inference on a 40gb GPU?

1. read dataset from huggingface
2. design chat template
3. format dataset with chat template
    how does llama.cpp receive the data?
    how to send system prompt just once?
4. start benchmark: split batch of data to multiple requests and send them to llama.cpp server
   pass bs np as params to the script
"""

import json
import pandas as pd
from torch.utils.data import DataLoader
import time
from datasets import load_dataset
import asyncio, aiohttp
import os

# read args from cli
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="ganler/code-r1-12k")
parser.add_argument("--bs", type=int, default=1)
parser.add_argument("--np", type=int, default=1)
parser.add_argument("--data_path", type=str, default="data")
parser.add_argument("--output_path", type=str, default="output")
parser.add_argument("--max_tokens", type=int, default=100)
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--top_p", type=float, default=0.9)
parser.add_argument("--top_k", type=int, default=0)
parser.add_argument("--max_new_tokens", type=int, default=100)


def load_hf_dataset(dataset_name, split="train"):
    dataset = load_dataset(dataset_name, split=split)
    return dataset


def make_map_fn(split):
    def process_fn(example, idx):
        question_raw = example.pop("prompt")

        return {"prompt": str(question_raw)}

    return process_fn


async def fetch(session, url, params):
    async with session.get(url, params=params) as r:
        return await r.text()


async def main():
    args = parser.parse_args()
    # load dataset

    # Check if train.parquet already exists
    parquet_path = os.path.join(args.data_path, "train.parquet")
    if os.path.exists(parquet_path):
        print(f"Found existing dataset at {parquet_path}, skipping dataset creation")
    else:
        dataset = load_hf_dataset(args.dataset_name, "train")
        # format dataset
        dataset = dataset.map(function=make_map_fn("train"), with_indices=True)

        print(f"Creating new dataset at {parquet_path}")
        dataset.to_parquet(parquet_path)

    df = pd.read_parquet(os.path.join(args.data_path, "train.parquet"))

    data = df.iloc[0 : args.bs]

    print(len(data))

    assert len(data) == args.bs

    assert args.bs % args.np == 0
    step = args.bs // args.np

    url = "http://localhost:8080/completion"

    async with aiohttp.ClientSession() as session:
        tasks = []

        for i in range(0, len(data), step):
            tasks.append(
                asyncio.create_task(
                    fetch(
                        session,
                        url,
                        {
                            "prompt": [
                                str(data.iloc[j]["prompt"]) for j in range(i, i + step)
                            ]
                        },
                    )
                )
            )

        responses = await asyncio.gather(*tasks)

        # save responses to file
        with open(os.path.join(args.output_path, "responses.json"), "w") as f:
            json.dump(responses, f)

        print("Got", len(responses), "responses")


if __name__ == "__main__":
    asyncio.run(main())
