import torch
import torch.nn as nn
import torch.optim as optim
import os
import csv
import numpy as np
from sklearn.metrics import confusion_matrix

from TrainingConfig import TrainingConfig
from LambdaScheduler import LambdaScheduler
from TotalLoss import TotalLoss


"""
logica:

# --------------------------------------------------------------------------
# Setup Inicial (O que vai no __init__):
# --------------------------------------------------------------------------
# A lógica agora é: Instanciar TUDO no __init__ (construtor) do StaticMaskTraining.
# Isso garante que:
# 1. model e mask_model são criados UMA VEZ.
# 2. optimizer é criado UMA VEZ (importante para manter o estado do treinamento/Adam).
# 3. total_loss_calculator e lambda_scheduler são criados UMA VEZ.
# -> Tudo isso vira atributo (self.model, self.optimizer, etc.).
# -> Evita recriar objetos caros e perder o progresso a cada epoch.

# --------------------------------------------------------------------------
# train_epoch(self, loader):
# --------------------------------------------------------------------------
# Responsabilidade: Foco total no cálculo (forward/backward).
# Argumentos: Só precisa de 'self' e 'loader'.
# Fluxo:
# 1. Pega os modelos e otimizador de 'self'.
# 2. Faz o forward pass: X_masked = self.mask_model(X) e y_pred = self.model(X_masked).
# 3. Loss: model_loss, mask_loss, total_loss = self.total_loss_calculator(...).
# 4. Ajuste: total_loss.backward() e self.optimizer.step().
# 5. CORREÇÃO: As perdas acumuladas (running_loss) precisam do '+=' para somar, 
#    e a média final é dividida por len(loader).

# --------------------------------------------------------------------------
# train(self, train_loader, val_loader):
# --------------------------------------------------------------------------
# Responsabilidade: Loop principal (Orquestração).
# 1. Faz o loop (for epoch in range...).
# 2. Chama self.train_epoch(train_loader).
# 3. Chama self.validate_epoch(val_loader).
# 4. Chama self.lambda_scheduler.adapt_lambda(avg_total_loss).
# 5. Chama self.save_checkpoint() com as informações do 'self' e da época.

# --------------------------------------------------------------------------
# Outros Componentes:
# --------------------------------------------------------------------------
# validate_epoch(): Metodo para pegar self.model e self.mask_model e rodar 
#                   o test_model (agora parte da classe ou auxiliar).
# DatasetDict: Serve para organizar vários datasets de forma limpa (auxiliar de setup).
# ModelTester: Classe separada, só para rodar testes finais no .pth salvo.
# TotalLoss: OK.
# TrainingConfig: OK, classe que tipa e centraliza as configs (LRs, épocas, etc.).
# Análise Estatística: Vai ser feita DEPOIS, usando os dados salvos nos checkpoints.
# training_loop.py: Vai chamar todas os métodos e fazer a parte do multithreading

"""

class StaticMaskTraining:
    def __init__(self, config: TrainingConfig):

        #Initializing the config file and the cuda device
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = config.seed

        
        #Initializing the dataloaders
        self.train_loader, self.val_loader, self.test_loader = self.config.get_dataloader(self.seed)


        #Initializing the models, Mask and Classifier
        self.model = self.config.model.to(self.device)
        self.mask_model = self.config.mask_model(shape=self.config.mask_shape).to(self.device)


        #Initializing the TotalLoss and Scheduler
        self.criterion = self.config.model_loss_function()

        self.lambda_scheduler = LambdaScheduler(init = self.config.lambda_init, 
                                                factor = self.config.lambda_factor, 
                                                patience = self.config.lambda_patience, 
                                                treshold = self.config.lambda_treshold)
        
        self.total_loss_calculator = TotalLoss(
            model_loss= self.criterion,
            mask_loss_function= self.config.mask_loss_function,
            lambda_scheduler=self.lambda_scheduler
        )

        #Initializing the Optimizer
        self.optimizer_class = self.config.optimizer_class
        self.optimizer_kwargs = {}

        if self.optimizer_class == optim.SGD:
            self.optimizer_kwargs['momentum'] = 0.9
        if self.optimizer_class == optim.Adam or self.optimizer_class == optim.AdamW:
            self.optimizer_kwargs['amsgrad'] = True

        self.optimizer = self.optimizer_class(
            [
            {'params': self.model.parameters(), 'lr': self.config.model_learning_rate},
            {'params': self.mask_model.parameters(), 'lr': self.config.mask_learning_rate}
        ],
        **self.optimizer_kwargs
    )

        #Initilizing the Checkpoints 
        self.training_id = self.config.training_id

        root_dir = getattr(self.config, 'root_dir_save', 'checkpoints')
        self.checkpoint_dir = os.path.abspath(os.path.join(root_dir, self.training_id))
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
        
    
    # pega tudo do config menos o scheduler
    def train_epoch(self):
        loader = self.train_loader

        running_model_loss = 0.0
        running_mask_loss = 0.0
        running_total_loss = 0.0

        for X, y in loader: 
            X = X.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            X_masked = self.mask_model(X)
            y_pred = self.model(X_masked)

            model_loss, mask_loss, total_loss = self.total_loss_calculator(
                pred=y_pred,
                target=y,
                mask_model=self.mask_model    
            )           

            total_loss.backward()
            self.optimizer.step()

            running_model_loss += model_loss.item()
            running_mask_loss += mask_loss.item()
            running_total_loss += total_loss.item()
        
        avg_model_loss = running_model_loss / len(loader)
        avg_mask_loss = running_mask_loss / len(loader)
        avg_total_loss = running_total_loss / len(loader)

        return avg_model_loss, avg_mask_loss, avg_total_loss

    @torch.no_grad()
    def validate_epoch(self):
        
        loader = self.val_loader
        all_targets = []
        all_predictions = []

        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                
                X_masked = self.mask_model(X)
                y_pred = self.model(X_masked)

                _, predicted = torch.max(y_pred, 1)

                all_targets.extend(y.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        cm = confusion_matrix(all_targets, all_predictions)
        accuracy = 100 * cm.diagonal().sum() / cm.sum()


        return cm, accuracy

    def save_checkpoint(self, epoch, cm, avg_total_loss, avg_model_loss, avg_mask_loss, accuracy):
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'mask_state_dict': self.mask_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_loss': avg_total_loss,
            'model_loss': avg_model_loss,
            'mask_loss': avg_mask_loss,
            'accuracy': accuracy,
            'cm': cm,
            'lambda': self.lambda_scheduler.lbd
        }
        torch.save(checkpoint, f'{self.checkpoint_dir}/checkpoint_epoch_{epoch + 1}.pt')

    def log_training(self, epoch, avg_model_loss, avg_mask_loss, avg_total_loss, accuracy):
        log_file = os.path.join(self.checkpoint_dir, 'training_log.csv')
        
        # 2. Definição dos Dados e Cabeçalho
        
        data = [
            epoch+1,
            avg_total_loss,
            avg_model_loss,
            avg_mask_loss,
            accuracy, # Acurácia já calculada
            self.lambda_scheduler.lbd,
            self.lambda_scheduler.count,
        ]
        
        header = [
            'epoch', 
            'total_loss', 
            'model_loss', 
            'mask_loss', 
            'val_accuracy', 
            'lambda_value',
            'lambda_patience_count',
        ]

        # 3. Escrita no CSV
        file_exists = os.path.exists(log_file)
        
        # Abrimos o arquivo para anexar ('a')
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # Escreve o cabeçalho apenas se o arquivo for novo
            if not file_exists:
                writer.writerow(header)
            
            # Escreve os dados da época
            writer.writerow(data)

    def train(self):

        print(f"Starting training ({self.config.n_epochs} epochs) in {self.device}.")
        print(f"Checkpoints will be saved in: {self.checkpoint_dir}")
        
        for epoch in range(self.config.n_epochs):
            print(f"\nEpoch: {epoch+1}/{self.config.n_epochs}")

            #Training
            self.model.train()
            self.mask_model.train()

            avg_model_loss, avg_mask_loss, avg_total_loss = self.train_epoch()

            self.lambda_scheduler.adapt_lambda(avg_total_loss)

            #Evaluating
            self.model.eval()
            self.mask_model.eval()
            
            cm_array, accuracy = self.validate_epoch()

            #Log
            print(f"\nTotal Loss: {avg_total_loss:.4f}, Model Loss: {avg_model_loss:.4f}, Mask Loss: {avg_mask_loss:.4f}"
                f"\nLambda: {self.lambda_scheduler.lbd:.4f}, Lambda Patience Count: {self.lambda_scheduler.count}"
                f"\nAccuracy: {accuracy:.4f}") 
            
            self.log_training(epoch, avg_model_loss, avg_mask_loss, avg_total_loss, accuracy)

            self.save_checkpoint(epoch, cm_array, avg_total_loss, avg_model_loss, avg_mask_loss, accuracy)


        print("Training finished.")



    