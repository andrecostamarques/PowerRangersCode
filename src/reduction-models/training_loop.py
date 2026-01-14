# codigo para o treinamento e testes de modelos
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix
import numpy as np

# --- GERENCIAMENTO DE CAMINHOS (PATH MANAGEMENT) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, '..'))
root_path = os.path.abspath(os.path.join(current_dir, '..', '..'))

for path in [os.path.join(src_path, 'utils'), os.path.join(src_path, 'models'), os.path.join(root_path, 'notebooks')]:
    if path not in sys.path: sys.path.append(path)

import NotebookUtils as nu
import DatasetsDict as dd

from ResNet20 import resnet20 
from LeNet256 import LeNet5_256

def load_frozen_mask(checkpoint_id, epoch, root_dir, device):
    """
    Loads the mask model architecture from Epoch 1 and applies weights from the target epoch.
    Ensures the model is fully frozen and in evaluation mode.
    """
    # Define os caminhos baseados na estrutura de pastas
    exp_dir = os.path.join(root_dir, checkpoint_id)
    path_epoch_1 = os.path.abspath(os.path.join(exp_dir, 'checkpoint_epoch_1.pt'))
    path_target = os.path.abspath(os.path.join(exp_dir, f'checkpoint_epoch_{epoch}.pt'))

    if not os.path.exists(path_epoch_1) or not os.path.exists(path_target):
        raise FileNotFoundError(f"Checkpoints not found in: {exp_dir}")

    # 1. Carrega o objeto (arquitetura) da Época 1
    # weights_only=False é necessário para recuperar o objeto 'mask_model_obj'
    checkpoint_e1 = torch.load(path_epoch_1, map_location=device, weights_only=False)
    mask_model = checkpoint_e1['mask_model_obj']

    # 2. Carrega os pesos (state_dict) da época alvo
    checkpoint_target = torch.load(path_target, map_location=device, weights_only=False)
    mask_model.load_state_dict(checkpoint_target['mask_state_dict'])

    # 3. Blindagem Total
    mask_model.to(device)
    mask_model.eval() # Trava BatchNorm/Dropout
    for param in mask_model.parameters():
        param.requires_grad = False # Desativa gradientes

    return mask_model


# --- HYPERPARAMETERS & SETTINGS ---
SETTINGS = {
    "dataset_name": "eurosat",
    "batch_size": 128,
    "val_ratio": 0.1,
    "train_epochs": 300,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "seed": 42,
    
    # --- NOVA FLAG ---
    "masked": False, # Se True, aplica a máscara. Se False, treina o modelo puro.
    
    # Modelo e Perda
    "model_class": LeNet5_256(),
    "criterion": nn.NLLLoss(),
    
    # Otimizador
    "optimizer_class": optim.Adam,
    "optimizer_kwargs": {"lr": 1e-3},
    
    # Máscara Blindada
    "mask_checkpoint_id": "galaxy10_resnet34_masked",
    "mask_epoch": 160,
    "root_checkpoints": os.path.join(root_path, "checkpoints"),
    "save_dir": os.path.join(root_path, "checkpoints", "eurosat_lenet_unmasked")
}

os.makedirs(SETTINGS["save_dir"], exist_ok=True)
utils = nu.NotebookUtils()

# --- 1. DATASET LOADING (WITH TRANSFORMS & SPLIT) ---
db = dd.DatasetDict()

# O get() já retorna os datasets instanciados com as transformações corretas
ds_list, _, _ = db.get(SETTINGS["dataset_name"])
train_ds_full = ds_list[0]

# Configuração do Generator para Reprodutibilidade
g = torch.Generator('cpu').manual_seed(SETTINGS["seed"])

# Cálculo do Split
val_size = int(len(train_ds_full) * SETTINGS["val_ratio"])
train_size = len(train_ds_full) - val_size

# Split determinístico usando o generator
train_ds, val_ds = random_split(
    train_ds_full, 
    [train_size, val_size],
    generator=g
)

# Dataloaders com Generator
# O generator aqui garante que o shuffle (embaralhamento) seja controlado pela seed
train_loader = DataLoader(
    train_ds, 
    batch_size=SETTINGS["batch_size"], 
    shuffle=True, 
    generator=g
)

val_loader = DataLoader(
    val_ds, 
    batch_size=SETTINGS["batch_size"], 
    shuffle=False
)

print(f"✅ Data Ready (Seed: {SETTINGS['seed']})")
print(f"   - Training: {train_size} samples")
print(f"   - Validation: {val_size} samples")

# --- 2. CARREGAMENTO CONDICIONAL DA MÁSCARA ---
mask_model = None
if SETTINGS["masked"]:
    mask_model = load_frozen_mask(
        checkpoint_id=SETTINGS["mask_checkpoint_id"],
        epoch=SETTINGS["mask_epoch"],
        root_dir=SETTINGS["root_checkpoints"],
        device=SETTINGS["device"]
    )
    print(f"✅ Mask loaded and frozen from: {SETTINGS['mask_checkpoint_id']} (Epoch {SETTINGS['mask_epoch']})")
else:
    print("ℹ️ Mode: UNMASKED. Training directly on raw dataset images.")

# --- 3. INICIALIZAÇÃO DINÂMICA DO OTIMIZADOR ---
model_classifier = SETTINGS["model_class"].to(SETTINGS["device"])

# Aplicação das condições solicitadas para o dicionário de argumentos
opt_class = SETTINGS["optimizer_class"]
opt_kwargs = SETTINGS["optimizer_kwargs"].copy()

if opt_class == optim.SGD:
    opt_kwargs['momentum'] = 0.9
if opt_class in [optim.Adam, optim.AdamW]:
    opt_kwargs['amsgrad'] = True

optimizer = opt_class(model_classifier.parameters(), **opt_kwargs)
criterion = SETTINGS["criterion"]

# --- 4. FUNÇÕES DE SUPORTE ADAPTADAS ---
@torch.no_grad()
def validate(model, mask_model, loader, device, use_mask):
    model.eval()
    if mask_model: mask_model.eval()
    
    all_targets, all_predictions = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        
        # Lógica Condicional de Validação
        if use_mask and mask_model:
            X = mask_model(X)
            
        outputs = model(X)
        _, predicted = torch.max(outputs, 1)
        all_targets.extend(y.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
        
    cm = confusion_matrix(all_targets, all_predictions)
    accuracy = 100 * cm.diagonal().sum() / cm.sum()
    return cm, accuracy

def save_checkpoint(epoch, model, mask_model, optimizer, accuracy, cm, save_path):
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy,
        'cm': cm,
        'mask_source': SETTINGS["mask_checkpoint_id"] if SETTINGS["masked"] else "None",
        'is_masked_training': SETTINGS["masked"]
    }
    if epoch == 0:
        checkpoint['model_obj'] = model
        if SETTINGS["masked"]:
            checkpoint['mask_model_obj'] = mask_model 
            
    torch.save(checkpoint, os.path.join(save_path, f'checkpoint_epoch_{epoch+1}.pt'))

# --- 5. LOOP DE TREINAMENTO ADAPTADO ---
mode_str = "COM MÁSCARA" if SETTINGS["masked"] else "PURO (UNMASKED)"
print(f"🚀 Treino Iniciado: {mode_str}")

for epoch in range(SETTINGS["train_epochs"]):
    model_classifier.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(SETTINGS["device"]), labels.to(SETTINGS["device"])
        
        # Aplicação condicional da máscara
        if SETTINGS["masked"] and mask_model:
            with torch.no_grad():
                images = mask_model(images)
        
        optimizer.zero_grad()
        outputs = model_classifier(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Validação e Checkpoint passandro a flag masked
    val_cm, val_acc = validate(
        model_classifier, 
        mask_model, 
        val_loader, 
        SETTINGS["device"], 
        SETTINGS["masked"]
    )
    
    save_checkpoint(epoch, model_classifier, mask_model, optimizer, val_acc, val_cm, SETTINGS["save_dir"])

    print(f"Época [{epoch+1}/{SETTINGS['train_epochs']}] - Loss: {running_loss/len(train_loader):.4f} - Val Acc: {val_acc:.2f}%")