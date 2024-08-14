from train import train_func



"""
Com essa função é possível treinar modelos em sequência dentro de um loop.
A ideia é usar listas e dicionários para armazenar os parâmetros de cada modelo.

A função de treino pode receber uma lista de parâmetros que representa os modelos
a serem treinados. Todos devem ter um respectivo nome para serem salvos, batch_size,
cap e flag do treinamento supervisionado    
"""


def main():

    EPOCAS = 50
    BATCH_SIZE = 8
    SUPERVISED = True
    FREEZE = False
    
    list_of_datas = ['f3', 'parihaka']
    list_of_pretrains = ['f3', 'parihaka', 'COCO', 'IMAGENET', 'both', 'sup']
    list_of_caps = [0.01, 0.1, 0.5, 1.0]
    
    REPETITIONS = 'V1'                          # qual repetição do experimento
    DOWNSTREAM_DATA = 'parihaka'                      # dados para a tarefa downstream 
    
    PRETRAIN_DATA = 'f3'                        # onde foi o pretreino
    MODE = 'imagenet'
    

    IMPORT_NAME = f'V1_E300_B32_S256_{PRETRAIN_DATA}' 
    
    print(f'''
        ******************** \n
        Início do treinamento no {PRETRAIN_DATA} com o pretreino no {DOWNSTREAM_DATA}\n
        Treinamento supervisionado normal\n
        ******************** \n
          ''')
    
    for cap in list_of_caps:
        
        print('********************')
        print(f'Rodando com cap {cap*100:.0f}%')
        print('********************')
        
        if MODE == 'supervised':
            save_name = f'{REPETITIONS}_sup_{DOWNSTREAM_DATA}_cap_{cap*100:.0f}%'
        elif MODE == 'byol':
            save_name = f'{REPETITIONS}_pre_{PRETRAIN_DATA}_train_{DOWNSTREAM_DATA}_cap_{cap*100:.0f}%'
        elif MODE == 'coco':
            save_name = f'{REPETITIONS}_pre_COCO_train_{DOWNSTREAM_DATA}_cap_{cap*100:.0f}%'
        elif MODE == 'imagenet':
            save_name = f'{REPETITIONS}_pre_IMAGENET_train_{DOWNSTREAM_DATA}_cap_{cap*100:.0f}%'
        
        print(f'******** Nome salvo: {save_name}')
        
        train_func(epocas=EPOCAS,
                   batch_size=BATCH_SIZE,
                   cap=cap,
                   import_name=IMPORT_NAME,
                   save_name=save_name,
                   supervised=SUPERVISED,
                   freeze=FREEZE,
                   downstream_data=DOWNSTREAM_DATA,
                   mode=MODE,
                   )
    
    print('main')
    
    
















if __name__ == "__main__":
    main()