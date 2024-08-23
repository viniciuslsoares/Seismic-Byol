from pretrain import pretrain_func



"""
Com essa função é possível treinar modelos em sequência dentro de um loop.
A ideia é usar listas e dicionários para armazenar os parâmetros de cada modelo.

A função de treino pode receber uma lista de parâmetros que representa os modelos
a serem treinados. Todos devem ter um respectivo nome para serem salvos, batch_size,
cap e flag do treinamento supervisionado    
"""



def main():
    
    NODE = 'node13[0]'
    REPORT_NAME = f'pretrain_{NODE}_run'

    report_path = 'reports/'

    EPOCAS = 300
    BATCH_SIZE = 32
    INPUT_SIZE = 256
    # REPETITIONS = 'V3'                          # qual repetição do experimento    
    
    list_of_repets = ['V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10']
    # list_of_datas = ['f3', 'seam_ai', 'both']
    list_of_datas = ['f3', 'seam_ai']
    # list_of_datas = ['both']
    # list_of_datas = ['seam_ai', 'both']
    
    # Três GPUs para rodar os experimentos
    # 4090 -> todos os f3
    # A6000 -> f3, parihaka e both no parihaka
    # A6000 -> coco, imagenet e sup no parihaka
    
    with open(report_path + f'{REPORT_NAME}.txt', 'w') as f:
        f.write('Report of the pretraining\n')
        f.write('---------------------------------------\n')
        f.write(f'Datas being used: {list_of_datas}\n')
        f.write(f'Node: {NODE}\n')
        f.write('---------------------------------------\n')
    
    for repetition in list_of_repets:
        for data in list_of_datas:
            
            print(30*'*-')
            print(f'Running with data {data}. ')
            print(30*'*-')
            
            with open(report_path + f'{REPORT_NAME}.txt', 'a') as f:
                f.write(30*'*-' + '\n')
                f.write(f'------------------ Pretraining on {data} ------------------\n')
            
            save_name = f'{repetition}_E{EPOCAS}_B{BATCH_SIZE}_S{INPUT_SIZE}_{data}'
            
            pretrain_func(epocas=EPOCAS,
                        batch_size=BATCH_SIZE,
                        input_size=INPUT_SIZE,
                        repetition=repetition,
                        save_name=save_name,
                        data=data
                        )

            with open(report_path + f'{REPORT_NAME}.txt', 'a') as f:
                f.write(f'------------------ Pretrain finished ------------------\n')
                

if __name__ == "__main__":
    main()