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
    parser.add_argument("--regex", type=str, default="(a b + b a) (a + b b + c)* (a c + b a)")           # (a(a)*b)* or (a b + b a) (a + b b + c)* (a c + b a)
    parser.add_argument("--max_length", type=int, default=8)
    parser.add_argument("--test_max_length", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_train_str_per_ce", type=int, default=3)
    parser.add_argument("--mode_train_str_from_ce", type=str, default="dfa_state",
                        choices=["dfa_state", "random", "repeat"])
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--epochs_per_round", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--mode", type=str, default="normal_CEs",
                        choices=["50_50_CEs", "100_CEs", "random_CEs", "normal_CEs"])
    parser.add_argument("--preheat", default=False, action="store_true")
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

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

    if args.preheat:
        x = task.generate_random_strings_balanced(128 , args.max_length)
        y = [int(task.accepts(ex)) for ex in x]
        losses = learner.train(
            x, y, 
            epochs=args.epochs_per_round,
            lr=args.lr,
            batch_size=args.batch_size,
            round=0
        )

        eval = teacher.judge(
            classifier=learner.classify,
            n=args.batch_size * 32,
            batch_size=args.batch_size,
            seq_len=args.test_max_length,
        )
        print(f"Pretrained on {len(x)} supervised examples, Eval Acc: {eval}")

    agg_losses, num_samples, accs = [], [], []
    num_train_samples, num_train_pos_sam = 0, 0
    for epoch in tqdm(range(args.rounds)):
        if args.mode == "50_50_CEs":
            exs = learner.generate_examples(
                n=args.batch_size, 
                batch_size=args.batch_size, 
                target=torch.tensor([0.5, 0.5]),
                seq_len=args.max_length,
                round=epoch,
                verbose=False
            )
            preds = learner.classify(exs)
            neg_ex = [ex for ex, pred in zip(exs, preds) if pred == 0]
            pos_ex = [ex for ex, pred in zip(exs, preds) if pred == 1]
        elif args.mode == "100_CEs":
            neg_ex = learner.generate_examples(
                n=args.batch_size, 
                batch_size=args.batch_size, 
                target=torch.tensor([1.0, 0.0]),
                seq_len=args.max_length,
                round=epoch,
                verbose=False
            )
            pos_ex = learner.generate_examples(
                n=args.batch_size, 
                batch_size=args.batch_size, 
                target=torch.tensor([0.0, 1.0]),
                seq_len=args.max_length,
                round=epoch,
                verbose=False
            )
            neg_pred = learner.classify(neg_ex)
            pos_pred = learner.classify(pos_ex)
            neg_ex = [ex for ex, pred in zip(neg_ex, neg_pred) if pred == 0]
            pos_ex = [ex for ex, pred in zip(pos_ex, pos_pred) if pred == 1]
        elif args.mode == "random_CEs":
            neg_ex, pos_ex = learner.generate_examples_from_random(
                n=args.batch_size, 
                seq_len=args.max_length,
                round=epoch
            )
        elif args.mode == "normal_CEs":
            tn = np.clip(np.random.normal(loc=0.5, scale=0.1), 0.0, 1.0)
            tp = 1.0 - tn
            exs = learner.generate_examples(
                n=args.batch_size, 
                batch_size=args.batch_size, 
                target=torch.tensor([tn, tp], dtype=torch.float32),
                seq_len=args.max_length,
                round=epoch,
                verbose=False
            )
            preds = learner.classify(exs)
            neg_ex = [ex for ex, pred in zip(exs, preds) if pred == 0]
            pos_ex = [ex for ex, pred in zip(exs, preds) if pred == 1]

        print(f"Epoch: {epoch}")
        print("Negative Examples")
        print(neg_ex)
        print(" ".join(["Pos" if learner.classify([ex])[0] == 1 else "Neg" for ex in neg_ex]))
        print("Positive Examples")
        print(pos_ex)
        print(" ".join(["Pos" if learner.classify([ex])[0] == 1 else "Neg" for ex in pos_ex]))

        ce_str, ce_y = teacher.generate_counterexamples(
            args.num_train_str_per_ce, 
            neg_ex, 
            pos_ex,
            mode=args.mode_train_str_from_ce
        )
        print("Counterexamples")
        print(ce_str)
        print(" ".join(["Pos" if y == 1 else "Neg" for y in ce_y]))
        if ce_str == [] and ce_y == []:
            print(f"Round {epoch}: No counterexamples found, skipped.")
        else:
            # num_train_samples += len(neg_ex) + len(pos_ex)
            num_train_samples += len(ce_str)
            num_train_pos_sam += np.sum(np.array(ce_y) == 1)

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
            n=args.batch_size * 32,
            batch_size=args.batch_size,
            seq_len=args.test_max_length,
        )
        num_samples.append(num_train_samples)
        accs.append(eval)
        print(f"Accuracy at epoch {epoch}: {eval}, total training samples: {num_train_samples}")

    print(f"Pos train / Tot train = {num_train_pos_sam} / {num_train_samples}")

    plot_loss_curve(agg_losses, "loss_curves", "Overall_Train_Losses")
    plot_accuracy_curve(num_samples, accs, "accuracy_curves", 
                        f"Regex={args.regex}-mode={args.mode}-train_length={args.max_length}-test_length={args.test_max_length}-num_aug={args.num_train_str_per_ce}-aug_strategy={args.mode_train_str_from_ce}-epochs_per_round={args.epochs_per_round}")