import torch
import torch.nn as nn

class TemperatureScalar(nn.Module):
    def __init__(self, init_temp=1.5):
        """
        Temperature scaling calibrator.
        Takes logit predictions and applies a learned temperature scale T.
        Calibrated Probability = sigmoid(logit / T)
        """
        super().__init__()
        # Must be single parameter that is strictly positive
        self.temperature = nn.Parameter(torch.ones(1) * init_temp)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, 1) or scalar.
        Returns: (B, 1) Calibrated probabilities.
        """
        t = torch.clamp(self.temperature, min=1e-3) # Prevent Div by Zero
        scaled_logits = logits / t
        return torch.sigmoid(scaled_logits)

    def calibrate(self, val_logits: torch.Tensor, val_labels: torch.Tensor, epochs=50, lr=0.01):
        """
        Fits temperature parameter to validation dataset to minimize Negative Log Likelihood.
        """
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=epochs)
        criterion = nn.BCEWithLogitsLoss()

        def eval_loss():
            optimizer.zero_grad()
            t = torch.clamp(self.temperature, min=1e-3)
            scaled = val_logits / t
            loss = criterion(scaled, val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        return self.temperature.item()
