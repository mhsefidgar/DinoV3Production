import torch
import torch.nn as nn
import torch.nn.functional as F

class Distiller(nn.Module):
    """
    Knowledge Distillation Wrapper for DINOv3.
    Distills knowledge from a heavy Teacher model (e.g., ViT-Large) to a Student model (e.g., ViT-Small).
    """
    def __init__(self, teacher, student, alpha=0.5, temperature=2.0):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.alpha = alpha
        self.temperature = temperature
        
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

    def forward_loss(self, images, labels=None):
        """
        Computes the distillation loss.
        """
        with torch.no_grad():
            teacher_logits = self.teacher(images)
        
        student_logits = self.student(images)
        
        # Soft-target distillation loss (KL Divergence)
        distill_loss = nn.KLDivLoss(reduction="batchmean")(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1)
        ) * (self.temperature ** 2)
        
        # Original task loss (if labels provided)
        task_loss = 0.0
        if labels is not None:
             task_loss = F.cross_entropy(student_logits, labels)
             
        total_loss = self.alpha * distill_loss + (1 - self.alpha) * task_loss
        return total_loss, student_logits

    def train_step(self, optimizer, images, labels=None):
        """
        Performs a single training step.
        """
        optimizer.zero_grad()
        loss, _ = self.forward_loss(images, labels)
        loss.backward()
        optimizer.step()
        return loss.item()
