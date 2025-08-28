import torch
import argparse
import random
import numpy as np
from tqdm import tqdm

from modeling.RNN import RNN
from tasks.rl import SimplyRegularLanguage
from learner import Learner
from teacher import Teacher
from curve import plot_loss_curve, plot_accuracy_curve

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_dfa_transitions(dfa):
    print("All states:")
    for state in dfa.states:
        print(f"  {state}")

    print("\nStart state:")
    print(f"  {dfa.start_state}")

    print("\nAccept states:")
    for state in dfa.final_states:
        print(f"  {state}")

    print("\nAll transitions:")
    for state in dfa.states:
        for symbol in dfa.symbols:
            try:
                target = dfa._transition_function(state, symbol)
                print(f"  {state} --{symbol}--> {target}")
            except KeyError:
                continue  # No transition defined

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--regex", nargs="+", default=["b(a(a)*b)*", "(b(b)*a)*b", "b(ab)*", "(a+bab)*", "(b+aba)*"])  # bababab...
    parser.add_argument("--max_length", type=int, default=8)
    parser.add_argument("--test_max_length", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--epochs_per_round", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    tasks = []
    num_alphabets = 0
    for r in args.regex:
        tasks.append(SimplyRegularLanguage(r, args.max_length))
        num_alphabets = max(num_alphabets, tasks[-1].num_alphabets)

    model = RNN(
        input_dim=num_alphabets,
        hidden_dim=args.hidden_dim, 
        output_dim=5, 
        num_layers=args.num_layers,
        device=device
    )
    learner = Learner(model, None)

    agg_losses, num_samples, accs = [], [], []
    num_train_samples, num_train_pos_sam = 0, 0
    for epoch in tqdm(range(args.rounds)):
        strs = tasks[0].generate_random_strings_balanced(
            n=args.batch_size, 
            seq_len=args.max_length
        )
        ys = []
        for str in strs:
            ct = 0
            for task in tasks:
                ct += int(task.accepts(str))
            ys.append(ct)
        num_train_samples += len(strs)
        
        losses = learner.train(
            strs, ys, 
            epochs=args.epochs_per_round,
            lr=args.lr,
            batch_size=args.batch_size,
            round=epoch
        )
        agg_losses += losses

        # Tests
        inputs = tasks[0].generate_random_strings_balanced(
            n=args.batch_size * 32, 
            seq_len=args.test_max_length
        )
        labels = []
        for str in inputs:
            ct = 0
            for task in tasks:
                ct += int(task.accepts(str))
            labels.append(int(ct == len(tasks)))
        pred = learner.classifier(inputs, args.batch_size)
        eval = sum([int(x == y) for x, y in zip(pred, labels)]) / len(pred)

        num_samples.append(num_train_samples)
        accs.append(eval)
        print(f"Accuracy at epoch {epoch}: {eval}, total training samples: {num_train_samples}")

    plot_loss_curve(agg_losses, "loss_curves", "Overall_Train_Losses")
    plot_accuracy_curve(num_samples, accs, "accuracy_curves", 
                        f"Mixed-train_length={args.max_length}-test_length={args.test_max_length}-epochs_per_round={args.epochs_per_round}")
