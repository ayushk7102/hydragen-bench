import argparse
from transformers import AutoTokenizer
from pathlib import Path
from typing import Optional, List
import torch

from hydragen.utils import dtype_map, maybe_init_dist, local_print
from hydragen.tp import from_pretrained_tp
from hydragen.llama import HydragenLlamaForCausalLM


def main(args):
    rank = maybe_init_dist()
    use_tp = rank is not None

    split_prompts = [prompt.split("|") for prompt in args.prompts]

    for i in range(len(split_prompts) - 1):
        assert (
            len(split_prompts[i + 1]) % len(split_prompts[i]) == 0
        ), "Number of prompts in each level must be evenly divided by number of prompts in previous level"

    dtype = dtype_map[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_name)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "right"

    local_print("Loading model...")
    if use_tp:
        assert args.tp_path is not None
        model = from_pretrained_tp(args.pretrained_name, args.tp_path, dtype)
    else:
        model = HydragenLlamaForCausalLM.from_pretrained(
            args.pretrained_name, torch_dtype=dtype, device_map=args.device
        )
    local_print("Done loading model!")

    torch.manual_seed(args.seed)

    def get_model_input(prompts: List[str], add_special_tokens: bool):
        encoded_prompts = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=add_special_tokens,
        )
        prompt_ids = encoded_prompts["input_ids"].to(args.device)
        prompt_attention_mask = encoded_prompts["attention_mask"].to(args.device)
        return prompt_ids, prompt_attention_mask

    tokenized_prompts = [
        get_model_input(prompt, add_special_tokens=(i == 0))
        for i, prompt in enumerate(split_prompts)
    ]

    input_ids = [prompt[0] for prompt in tokenized_prompts]
    sequence_lengths = [prompt[1].sum(1) for prompt in tokenized_prompts]

    if args.num_return_sequences > 1:
        shared_batch_sizes = [ids.shape[0] for ids in input_ids]
        shared_seq_lengths = [ids.shape[1] for ids in input_ids]
    else:
        shared_batch_sizes = [ids.shape[0] for ids in input_ids[:-1]]
        shared_seq_lengths = [ids.shape[1] for ids in input_ids[:-1]]

    unique_batch_size = input_ids[-1].shape[0] * args.num_return_sequences
    unique_seq_len = args.max_new_tokens

    model.setup_caches(
        max_unique_batch_size=unique_batch_size,
        max_unique_seq_length=unique_seq_len,
        max_shared_batch_sizes=shared_batch_sizes,
        max_shared_seq_lengths=shared_seq_lengths,
    )

    model.graph(args.graph)

    new_ids = model.generate(
        input_ids=input_ids,
        seq_lens=sequence_lengths,
        num_return_sequences=args.num_return_sequences,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    local_print("Completions:")

    local_print(
        tokenizer.batch_decode(
            new_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hydragen model inference")
    parser.add_argument("--pretrained-name", type=str, default="princeton-nlp/Sheared-LLaMA-1.3B")
    parser.add_argument("--prompts", nargs="+", default=[
        "Harry Potter is a character.",
        "He is a wizard.|His best friend is Hermione.",
        "He went to school at|He is the main character in|She is known for her|Played by the actress"
    ])
    parser.add_argument("--num-return-sequences", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--graph", type=bool, default=True)
    parser.add_argument("--tp-path", type=Path, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)
