import torch
import torch.nn.functional as F
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from curve import plot_loss_curve

class Learner:
    def __init__(self, model, task):
        self.model = model
        self.task = task

        self.device = next(model.parameters()).device

    def onehotreg(self, x, mask=None):
        x_prob = F.softmax(x, dim=-1)
        max_probs = x_prob.max(dim=-1)[0]
        if mask is not None:
            return ((1.0 - max_probs) * mask).mean()
        else:
            return (1.0 - max_probs).mean()

    def generate_examples(self, n, batch_size, target, seq_len, round=0, verbose=False):
        self.model.train()
        for p in self.model.parameters():
            p.requires_grad = False
        criterion = torch.nn.MSELoss()
        reg_weight = 0.3
        
        examples = []
        for i in range(0, n, batch_size):
            num = min(i+batch_size, n) - i
            lengths = torch.randint(1, seq_len + 1, (num, ), device=self.device)
            x = torch.randn(num, seq_len + 1, self.model.input_dim, device=self.device)
            for r in range(num):
                x[r, lengths[r], :] = 1.0
                x[r, lengths[r]+1:, :] = 0.0
            x.requires_grad = True
            y = target.unsqueeze(0).repeat(num, 1).to(self.device)
            mask = torch.zeros(x.shape[:2], device=self.device)
            for r in range(num):
                mask[r, :lengths[r]] = 1.0
            
            optimizer = torch.optim.Adam([x], lr=0.01)
            losses = []
            patience, patience_counter = 10, 0
            delta = 0.001
            best_loss = float('inf')
            for step in range(1024):
                optimizer.zero_grad()
                x_softmaxed = torch.where(
                    mask.unsqueeze(-1) == 1.0,
                    F.softmax(x, dim=-1),
                    x
                )
                logits = self.model(x_softmaxed, lengths)

                loss_cls = criterion(logits, y)
                loss_reg = self.onehotreg(x, mask=mask)
                loss = loss_cls + reg_weight * loss_reg
                losses.append(loss.item())

                loss.backward()
                x.grad *= mask.unsqueeze(-1)
                optimizer.step()

                if loss.item() < best_loss - delta:
                    best_loss = loss.item()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {step}. Loss did not improve for {patience} epochs.')
                    break

            print(f"Generate examples Step {step}, Loss {np.mean(losses)}")
            # if verbose:
                # plot_loss_curve(losses, "loss_curves", f"Target#{target}-Round#{round}")
        
            x = x.detach()
            for r, sample in enumerate(x):
                chars = F.softmax(sample, dim=-1).max(dim=-1)[1][:lengths[r]]
                if verbose:
                    print(f"Round {round}, Example {i+r}, input logits: {F.softmax(sample, dim=-1)}")
                string = "".join([chr(c + ord('a')) for c in chars])
                examples.append(string)

            # Put the examples back and check the quality
            if verbose:
                self.model.eval()
                batch_x_tensor, lengths = self.task.to_tensor(examples)
                batch_x_tensor = batch_x_tensor.to(self.device)

                with torch.no_grad():
                    logits = self.model(batch_x_tensor, lengths)
                    print(f"Round {round}, output logits: {F.softmax(logits, dim=-1)}")
        
        return np.unique(examples).astype(str).tolist()
    
    def generate_examples_from_random(self, n, seq_len, round=0):
        strs = self.task.generate_random_strings(n, seq_len)
        preds = self.classify(strs)

        pos_str, neg_str = [], []
        for string, pred in zip(strs, preds):
            if pred == 0:
                neg_str.append(string)
            if pred == 1:
                pos_str.append(string)
        
        return np.unique(neg_str).astype(str).tolist(), np.unique(pos_str).astype(str).tolist()

    def train(self, x, y, epochs, lr, batch_size, round=0):
        self.model.train()
        for p in self.model.parameters():
            p.requires_grad = True

        class_weights = compute_class_weight(
            class_weight='balanced', 
            classes=np.unique(y),
            y=y
        )
        num_classes = self.task.num_categories
        weight_tensor = torch.zeros(num_classes).to(self.device)
        for i, class_id in enumerate(np.unique(y)):
            weight_tensor[class_id] = class_weights[i]
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor)

        for epoch in range(epochs):
            losses = []
            for i in range(0, len(x), batch_size):
                batch_x = x[i:min(i+batch_size, len(x))]
                batch_y = y[i:min(i+batch_size, len(x))]
                batch_x_tensor, lengths = self.task.to_tensor(batch_x)
                batch_x_tensor = batch_x_tensor.to(self.device)
                batch_y_tensor = torch.tensor(batch_y).to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_x_tensor, lengths)
                loss = criterion(outputs, batch_y_tensor)
                losses.append(loss.item())

                loss.backward()
                optimizer.step()
            print(f"Train Epoch {epoch}, Loss {np.mean(losses)}")

        return losses

    def classify(self, strings, batch_size=32):
        self.model.eval()
        res = []
        for i in range(0, len(strings), batch_size):
            batch_x = strings[i:min(i+batch_size, len(strings))]
            batch_x_tensor, lengths = self.task.to_tensor(batch_x)
            batch_x_tensor = batch_x_tensor.to(self.device)

            with torch.no_grad():
                logits = self.model(batch_x_tensor, lengths)
                pred = torch.argmax(logits, dim=1).detach()
            
            res += pred
        return res