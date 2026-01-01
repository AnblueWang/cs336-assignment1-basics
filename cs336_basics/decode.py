import torch
from cs336_basics.tokenizer.tokenizer import Tokenizer

from cs336_basics.model import transformer, functions
import argparse, numpy as np, logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("inference.log"), # Save to file
        logging.StreamHandler()          # Also print to console
    ]
)

def decode(user_prompt: str, max_generation_length: int = 100, temperature: float = 1.0, top_p: float = 1.0) -> str:
    parser = argparse.ArgumentParser(description="transformerlm from scratch")

    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--d_model", type=int, default=160)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=640)
    parser.add_argument("--use_cuda", type=bool, default=False)
    parser.add_argument("--max_seqlen", type=int, default=200)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--rope_theta", type=int, default=10000)
    parser.add_argument("--checkpoint_file", type=str, default="./data/checkpoint_step_1000.ckpt")
    parser.add_argument("--vocab_filepath", type=str, default="./data/vocab_output_10000.txt")
    parser.add_argument("--merges_filepath", type=str, default="./data/merges_output_10000.txt")
    parser.add_argument("--special_tokens", type=list[str], default=["<|endoftext|>"])
    parser.add_argument("--user_prompt", type=str, default=None)
    parser.add_argument("--max_generation_length", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_p", type=float, default=0.9)


    args = parser.parse_args()
    logger = logging.getLogger("training")
    logger.info(vars(args))

    if args.user_prompt is not None:
        user_prompt = args.user_prompt
    max_generation_length = args.max_generation_length
    temperature = args.temperature
    top_p = args.top_p

    # define model and optimizer
    model = transformer.TransformerLM(args.vocab_size, args.max_seqlen, args.num_layers, args.d_model, args.num_heads, args.d_ff, device="cuda:0" if args.use_cuda else "cpu")
    tokenizer = Tokenizer.from_files(Tokenizer,vocab_filepath=args.vocab_filepath, merges_filepath=args.merges_filepath, special_tokens=args.special_tokens)
    state_dict = torch.load(args.checkpoint_file)
    model.load_state_dict(state_dict["model"])
    input_tokens = torch.tensor(np.array([tokenizer.encode(user_prompt)],dtype=int)) #batch_size 1
    index = 0
    while index < max_generation_length and input_tokens[0][-1] != 0:
        output_logits = model.forward(input_tokens, args.rope_theta)
        sorted_logits, sorted_indices = torch.sort(output_logits[0,-1]/temperature, descending=True)
        cumulative_probs = torch.cumsum(functions.softmax(sorted_logits, di=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep at least one token
        # and to include the token that exactly crosses the p threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Zero out the logits of removed tokens by setting them to -inf
        # We apply the mask back to the original index positions
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        output_logits[0,-1][indices_to_remove] = float('-inf')

        probs = functions.softmax(output_logits[0,-1], di=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_tokens = torch.cat((input_tokens[0], torch.tensor([next_token])))
        input_tokens = input_tokens.unsqueeze(0)
        index += 1
    logger.info(f"The total compleletion has {index} generated tokens. put as a whole: {tokenizer.decode(input_tokens[0].tolist())}")

if __name__ == "__main__":
    decode("I have a good memory.")