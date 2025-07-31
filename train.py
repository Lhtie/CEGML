import torch
import argparse
import numpy as np
from tqdm import tqdm

from modeling.RNN import RNN
from tasks.rl import SimplyRegularLanguage
from learner import Learner
from teacher import Teacher
from curve import plot_loss_curve

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
    parser.add_argument("--regex", type=str, default="(a(a)*b)*")
    parser.add_argument("--max_length", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--epochs_per_round", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    task = SimplyRegularLanguage(args.regex, args.max_length)
    model = RNN(
        input_dim=task.num_alphabets, 
        hidden_dim=args.hidden_dim, 
        output_dim=2, 
        num_layers=args.num_layers,
        device=device
    )

    teacher = Teacher(task)
    learner = Learner(model, task)

    agg_losses = []
    for epoch in tqdm(range(args.rounds)):
        neg_ex = learner.generate_examples(
            n=args.batch_size, 
            batch_size=args.batch_size, 
            target=0,
            seq_len=args.max_length,
            round=epoch
        )
        neg_pred = learner.classify(neg_ex)
        pos_ex = learner.generate_examples(
            n=args.batch_size, 
            batch_size=args.batch_size, 
            target=1,
            seq_len=args.max_length,
            round=epoch
        )
        pos_pred = learner.classify(pos_ex)
        neg_ex = [ex for ex, pred in zip(neg_ex, neg_pred) if pred == 0]
        pos_ex = [ex for ex, pred in zip(pos_ex, pos_pred) if pred == 1]

        print(f"Epoch: {epoch}")
        print("Negative Examples")
        print(neg_ex)
        print(" ".join(["Pos" if learner.classify([ex])[0] == 1 else "Neg" for ex in neg_ex]))
        print("Positive Examples")
        print(pos_ex)
        print(" ".join(["Pos" if learner.classify([ex])[0] == 1 else "Neg" for ex in pos_ex]))

        ce_str, ce_y = teacher.generate_counterexamples(args.batch_size, neg_ex, pos_ex)
        print("Counterexamples")
        print(ce_str)
        print(" ".join(["Pos" if y == 1 else "Neg" for y in ce_y]))
        if ce_str == [] and ce_y == []:
            print(f"Round {epoch}: No counterexamples found, skipped.")
            continue
        # if len(np.unique(ce_y)) != task.num_categories:
        #     print(f"Round {epoch}: Not enough categories provided in training examples, skipped.")
        #     continue

        losses = learner.train(
            ce_str, ce_y, 
            epochs=args.epochs_per_round,
            lr=args.lr,
            batch_size=args.batch_size,
            round=epoch
        )
        agg_losses += losses
        eval = teacher.judge(
            classifier=learner.classify,
            n=args.batch_size * 8,
            batch_size=args.batch_size
        )
        print(f"Accuracy at epoch {epoch}: {eval}")

    plot_loss_curve(agg_losses, "loss_curves", "Overall_Train_Losses")
