# codigo para colocar todas as funções de utilidades para os testes de reduction-models
# funções que precisamos fazer:
# wrapper pra retornar só o string byte
# wrapper pra retornar a imagem mascarada


#informações:
# o wrapper tem que ser no datalodar mesmo, quando tudo tiver na gpu
# é mais rapido aplicar o modelo forward na gpu que no compose da cpu
# então é literamlente só fazer o training_loop no pelo mesmo 
# só aplicar a mascara, pegar o tensor resultante e treinar um modelo 
# ou pegar o tensor resultante e fazer flaten e treinar algum outro


"""
or images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)
    
    # Aplica a máscara sem rastrear gradientes
    with torch.no_grad():
        masked_images = mask_model(images)
    
    # Daqui para frente o gradiente volta a funcionar normalmente para o modelo principal
    outputs = main_model(masked_images)
    loss = criterion(outputs, labels)
    ...

ou usar 

mask_model.eval()
e o modelo de classificação ser 
model.train()


da pra usar todas as opções de congelamento também

# --- Setup ---
mask_model = SeuModeloDeMascara().to(device)
mask_model.eval()
for param in mask_model.parameters():
    param.requires_grad = False

optimizer = torch.optim.Adam(main_model.parameters(), lr=1e-3) # A máscara NÃO entra aqui

# --- Loop ---
for images, labels in train_loader:
    images = images.to(device)
    
    with torch.no_grad():
        images = mask_model(images) # Aplicação limpa e protegida
        
    # Treino normal do modelo principal
    outputs = main_model(images)
    ...

"""

class ReductionUtils:
    def __init__(self):
        pass

