from train import train_func



"""
Com essa função é possível treinar modelos em sequência dentro de um loop.
A ideia é usar listas e dicionários para armazenar os parâmetros de cada modelo.

A função de treino pode receber uma lista de parâmetros que representa os modelos
a serem treinados. Todos devem ter um respectivo nome para serem salvos, batch_size,
cap e flag do treinamento supervisionado    
"""


def main():
    
    NODE = 'node13[1]'
    REPORT_NAME = f'{NODE}_run'

    report_path = 'reports/'

    EPOCAS = 10
    BATCH_SIZE = 8
    FREEZE = False
    REPETITIONS = 'V1'                          # qual repetição do experimento    
    
    # list_of_datas = ['f3', 'parihaka']
    # list_of_datas = ['f3']
    list_of_datas = ['parihaka']
    
    # list_of_pretrains = ['f3', 'parihaka', 'COCO', 'IMAGENET', 'both', 'sup']
    # list_of_pretrains = ['f3', 'parihaka', 'both']
    list_of_pretrains = ['COCO', 'IMAGENET', 'sup']
    
    list_of_caps = [0.01, 0.1, 0.5, 1.0]
    
    # Três GPUs para rodas os experimentos
    # Posso separar os caps como 1, 10 e 50% e rodar em paralelo
    # com o de 100% OU posso separar os dados e rodar em paralelo
    # com todos os caps. Não sei qual o melhor a fazer.
    
    # 4090 -> todos os f3
    # A6000 -> f3, parihaka e both no parihaka
    # A6000 -> coco, imagenet e sup no parihaka
    
    with open(report_path + f'{REPORT_NAME}.txt', 'w') as f:
        f.write('Report of the training\n')
        f.write('---------------------------------------\n')
        f.write(f'Datas being used: {list_of_datas}\n')
        f.write(f'Pretrains models: {list_of_pretrains}\n')
        f.write(f'Caps: {list_of_caps}\n')
        f.write(f'Node: {NODE}\n')
        f.write('---------------------------------------\n')
    
    
    for pretrain in list_of_pretrains:
        for data in list_of_datas:
            for cap in list_of_caps:
                
                print(30*'*-')
                print(f'Running with data {data} and model pretrained in {pretrain} with cap {cap*100:.0f}%. ')
                print(30*'*-')
            
                if pretrain == 'f3' or pretrain == 'parihaka' or pretrain == 'both':
                    mode = 'byol'
                    supervised = False
                    import_name = f'{REPETITIONS}_E300_B32_S256_{pretrain}'
                    save_name = f'{REPETITIONS}_pre_{pretrain}_train_{data}_cap_{cap*100:.0f}%'
                
                elif pretrain == 'COCO':
                    mode = 'coco'
                    supervised = False
                    # Não importa, não será usado. Deve ser um import válido
                    import_name = f'{REPETITIONS}_E300_B32_S256_f3'         
                    save_name = f'{REPETITIONS}_pre_COCO_train_{data}_cap_{cap*100:.0f}%'
                
                elif pretrain == 'IMAGENET':
                    mode = 'imagenet'
                    supervised = False
                    # Não importa, não será usado. Deve ser um import válido
                    import_name = f'{REPETITIONS}_E300_B32_S256_f3'
                    save_name = f'{REPETITIONS}_pre_IMAGENET_train_{data}_cap_{cap*100:.0f}%'
                
                elif pretrain == 'sup':
                    mode = 'supervised'
                    supervised = True
                    # Não importa, não será usado. Deve ser um import válido
                    import_name = f'{REPETITIONS}_E300_B32_S256_f3'
                    save_name = f'{REPETITIONS}_sup_{data}_cap_{cap*100:.0f}%'
                    
                with open(report_path + f'{REPORT_NAME}.txt', 'a') as f:
                        f.write(f'------------------ Pre on {pretrain} and train on {data} ------------------\n')
                        f.write(f'Running with data {data} and model pretrained in {pretrain} with cap {cap*100:.0f}%. ')
                        f.write(f'Import name: {import_name}\n')
                        f.write(f'Save name: {save_name}\n')
                        f.write(f'Mode: {mode}; Cap: {cap*100:.0f}%; Repetitions: {REPETITIONS}\n')
                    
                train_func(
                    epocas=EPOCAS,
                    batch_size=BATCH_SIZE,
                    cap=cap,
                    import_name=import_name,
                    save_name=save_name,
                    supervised=supervised,
                    freeze=FREEZE,
                    downstream_data=data,
                    mode=mode,
                    repetition=REPETITIONS
                )
                    
                with open(report_path + f'{REPORT_NAME}.txt', 'a') as f:
                    f.write('---------------------------------------\n')
                    f.write('Treinamento finalizado\n')
                    f.write('---------------------------------------\n')
                        

if __name__ == "__main__":
    main()