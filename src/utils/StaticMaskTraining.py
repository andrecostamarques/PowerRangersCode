import torch

class StaticMaskTraining:
    def __init__(self, config):
        self.config = config

    def train_one_epoch(self):
        # Treinar o modelo a partir do datalodar de treino usando a loss total
        # Retorna a loss total, loss da mascara e loss do modelo
        pass

    def validate_epoch(self):
        # Pegar o modelo atual e passar os dados de validacao e calcula as metricas
        # Metricas: fp, fn, tp, tn
        # Retorna as metricas
        pass

    def train(self):
        # definir o loop a partir do numero de epocas, salva os checkpoints e as metricas de validacao, 
        pass

    def test(self, model_pth : string):
        # Carrega o pth, passa pelo dataloader de test, retornas a matriz de confuzao (scikit-learn)
        # obs: as metricas vao ser calculadas externamente (notebook ou outros arquivos)
        pass