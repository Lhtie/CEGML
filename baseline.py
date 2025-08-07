import torch
import argparse
import random
import numpy as np
from tqdm import tqdm

from modeling.RNN import RNN
from tasks.rl import SimplyRegularLanguage
from learner import Learner
from teacher import Teacher
from curve import plot_accuracy_curve

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--regex", type=str, default="(a(a)*b)*")
    parser.add_argument("--max_length", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=12)
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

    num_samples, accs = [], []
    num_train_samples = 0
    for epoch in tqdm(range(args.rounds)):
        ce_str = teacher._generate_random_ab_strings(args.batch_size, args.max_length)
        ce_y = [int(task.accepts(i)) for i in ce_str]

        learner.train(
            ce_str, ce_y, 
            epochs=3,
            lr=args.lr,
            batch_size=args.batch_size
        )
        num_train_samples += len(ce_str)

        if True:
            eval = teacher.judge(
                classifier=learner.classify,
                n=args.batch_size * 32,
                batch_size=args.batch_size
            )
            print(f"Accuracy at epoch {epoch}: {eval}")

            num_samples.append(num_train_samples)
            accs.append(eval)

    plot_accuracy_curve(num_samples, accs, "accuracy_curves", "Overall_Accuracy_baseline")