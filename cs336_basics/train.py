from cs336_basics.model import transformer, functions, adamw
import argparse, numpy as np, logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("train.log"), # Save to file
        logging.StreamHandler()          # Also print to console
    ]
)

def main():
    parser = argparse.ArgumentParser(description="transformerlm from scratch")

    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--d_model", type=int, default=160)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=640)
    parser.add_argument("--use_cuda", type=bool, default=False)
    parser.add_argument("--max_seqlen", type=int, default=200)
    parser.add_argument("--training_iterations", type=int, default=1000)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--training_data", type=str, default="")
    parser.add_argument("--valid_data", type=str, default="")
    parser.add_argument("--rope_theta", type=int, default=10000)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--max_learning_rate", type=float, default=0.01)
    parser.add_argument("--warmup_iters", type=int, default=100)
    parser.add_argument("--cosine_cycle_iters", type=int, default=500)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.99)
    parser.add_argument("--adam_weightdecay", type=float, default=0.01)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--grad_clipping_eps", type=float, default=1e-8)
    parser.add_argument("--checkpoint_file", type=str, default=None)

    args = parser.parse_args()
    logger = logging.getLogger("training")
    logger.info(vars(args))

    # define model and optimizer
    model = transformer.TransformerLM(args.vocab_size, args.max_seqlen, args.num_layers, args.d_model, args.num_heads, args.d_ff, device="cuda:0" if args.use_cuda else "cpu")
    optimizer = adamw.AdamW(model.parameters(), args.learning_rate, (args.adam_beta1, args.adam_beta2), eps=args.adam_eps, weight_decay=args.adam_weightdecay)
    step = 1
    if args.checkpoint_file != None:
        step = functions.load_checkpoint(args.checkpoint_file, model=model, optimizer=optimizer)

    # load data
    training_tokens = np.load(args.training_data, mmap_mode="r")
    valid_tokens = np.load(args.valid_data, mmap_mode="r")

    # start training
    while step < args.training_iterations:
        training_x, labels = functions.get_batch_input(training_tokens, args.batch_size, args.max_seqlen, "cuda:0" if args.use_cuda else "cpu")
        output = model.forward(training_x, args.rope_theta)
        loss = functions.cross_entropy(output, labels)
        loss.backward()
        functions.gradient_clipping(model.parameters(), args.max_grad_norm, eps=args.grad_clipping_eps)
        lr = functions.learning_rate_schedule(step, max_learning_rate=args.max_learning_rate, min_learning_rate=args.learning_rate, warmup_iters=args.warmup_iters, cosine_cycle_iters=args.cosine_cycle_iters)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        step += 1

        if step % 100 == 0:
            functions.save_checkpoint(model=model, optimizer=optimizer, iteration=step, out=f"./data/checkpoint_step_{step}.ckpt")
            valid_x, valid_labels = functions.get_batch_input(valid_tokens, args.batch_size, args.max_seqlen, "cuda:0" if args.use_cuda else "cpu")
            valid_output = model.forward(valid_x, args.rope_theta)
            valid_loss = functions.cross_entropy(valid_output, valid_labels)
            logger.info(f"Step {step}: Training loss {loss} and Valid loss {valid_loss}")


if __name__ == "__main__":
    
    main()