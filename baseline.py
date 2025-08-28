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
    parser.add_argument("--regex", type=str, default="(a(a)*b)*")        # (a b + b a) (a + b b + c)* (a c + b a)
    parser.add_argument("--max_length", type=int, default=8)
    parser.add_argument("--test_max_length", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--epochs_per_round", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--posrate", type=float, default=0.5)
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

    num_samples, accs = [], []
    num_train_samples = 0
    for epoch in tqdm(range(args.rounds)):
        # str = task.generate_random_strings_uniform(args.batch_size, args.max_length)
        # str = task.generate_random_strings_beta(
        #     args.batch_size,
        #     args.max_length,
        #     alpha=[2.0, 1.0]
        # )
        str = task.generate_random_strings_balanced(args.batch_size, args.max_length, rate=args.posrate)
        y = [int(task.accepts(i)) for i in str]
        print(str)

        learner.train(
            str, y, 
            epochs=3,
            lr=args.lr,
            batch_size=args.batch_size
        )
        num_train_samples += len(str)

        if True:
            eval = teacher.judge(
                classifier=learner.classify,
                n=args.batch_size * 32,
                batch_size=args.batch_size,
                seq_len=args.test_max_length,
            )
            print(f"Accuracy at epoch {epoch}: {eval}")

            num_samples.append(num_train_samples)
            accs.append(eval)

    plot_accuracy_curve(num_samples, accs, "accuracy_curves", 
                        f"Regex={args.regex}-mode=baseline-train_length={args.max_length}-test_length={args.test_max_length}-num_aug=1-aug_strategy=dfa_state-epochs_per_round={args.epochs_per_round}-posrate={args.posrate}")