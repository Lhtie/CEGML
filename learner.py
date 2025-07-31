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

    def onehotreg(self, x):
        x_prob = F.softmax(x, dim=-1)
        max_probs = x_prob.max(dim=-1)[0]
        return (1.0 - max_probs).mean()

    def generate_examples(self, n, batch_size, target, seq_len, round=0):
        self.model.train()
        for p in self.model.parameters():
            p.requires_grad = False
        criterion = torch.nn.CrossEntropyLoss()
        reg_weight = 0.5
        
        examples = []
        for i in range(0, n, batch_size):
            num = min(i+batch_size, n) - i
            x = torch.randn(num, seq_len + 1, self.model.input_dim, 
                            requires_grad=True, device=self.device)
            lengths = torch.full((num, ), seq_len, device=self.device)
            y = torch.full((num, ), target, device=self.device)
            
            optimizer = torch.optim.Adam([x], lr=0.03)
            losses = []
            patience, patience_counter = 10, 0
            delta = 0.001
            best_loss = float('inf')
            for step in range(1024):
                optimizer.zero_grad()
                logits = self.model(x, lengths)

                loss_cls = criterion(logits, y)
                loss_reg = self.onehotreg(x)
                loss = loss_cls + reg_weight * loss_reg
                losses.append(loss.item())

                loss.backward()
                optimizer.step()

                if loss.item() < best_loss - delta:
                    best_loss = loss.item()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {step}. Loss did not improve for {patience} epochs.')
                    break
                # if step % 10 == 0:
                    # print(f"Generate examples Step {step}, Loss {loss.item()}")
            print(f"Generate examples Step {step}, Loss {np.mean(losses)}")
            # plot_loss_curve(losses, "loss_curves", f"Target#{target}-Round#{round}")
        
            x = x.detach()
            for sample in x:
                chars = F.softmax(sample, dim=-1).max(dim=-1)[1]
                string = "".join([chr(c + ord('a')) for c in chars])
                examples.append(string)
        
        return examples
    
    def generate_examples_from_random(self, n, seq_len, round=0):
        strs = self.task.generate_random_strings(n, seq_len)
        preds = self.classify(strs)

        pos_str, neg_str = [], []
        for string, pred in zip(strs, preds):
            if pred == 0:
                neg_str.append(string)
            if pred == 1:
                pos_str.append(string)
        
        return neg_str, pos_str

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