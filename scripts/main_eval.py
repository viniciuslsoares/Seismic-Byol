from evaluate import eval_func



"""
Com essa função é possível avaliar modelos em sequência dentro de um loop.
A ideia é usar listas e dicionários para armazenar os parâmetros de cada modelo.    
"""


def main():
    
    REPORT_NAME = 'teste'

    report_path = 'reports/'
    
        
    list_of_datas = ['f3', 'parihaka']
    list_of_pretrains = ['f3']
    # list_of_pretrains = ['f3', 'parihaka', 'COCO', 'IMAGENET', 'both', 'sup']
    # list_of_pretrains = ['f3', 'parihaka', 'both']
    list_of_pretrains = ['COCO', 'IMAGENET', 'sup']
    list_of_caps = [0.01, 0.1, 0.5, 1.0]


    with open(report_path + f'{REPORT_NAME}.txt', 'w') as f:
        f.write('Report of the evaluation of the models\n')
        f.write('---------------------------------------\n')
        f.write(f'Datas being used: {list_of_datas}\n')
        f.write(f'Pretrains models: {list_of_pretrains}\n')
        f.write(f'Caps: {list_of_caps}\n')
        
    
    REPETITIONS = 'V1'                          # qual repetição do experimento
    
    
    for data in list_of_datas:
        for pretrain in list_of_pretrains:
            
            with open(report_path + f'{REPORT_NAME}.txt', 'a') as f:
                f.write(f'------------------ Pre on {pretrain} and train on {data} ------------------\n')
            
            for  cap in list_of_caps:

                print(30*'*-')
                print(f'''Running with data {data} and model pretrained in  {pretrain} with cap {cap*100:.0f}%. Evaluating test dataset''')
                print(30*'*-')

                if pretrain == 'f3' or pretrain == 'parihaka' or pretrain == 'both':
                    mode = 'byol'
                    import_name = f'{REPETITIONS}_pre_{pretrain}_train_{data}_cap_{cap*100:.0f}%'
                elif pretrain == 'COCO':
                    mode = 'coco'
                    import_name = f'{REPETITIONS}_pre_COCO_train_{data}_cap_{cap*100:.0f}%'
                elif pretrain == 'IMAGENET':
                    mode = 'imagenet'
                    import_name = f'{REPETITIONS}_pre_IMAGENET_train_{data}_cap_{cap*100:.0f}%'
                elif pretrain == 'sup':
                    mode = 'supervised'
                    import_name = f'{REPETITIONS}_sup_{data}_cap_{cap*100:.0f}%'


                iou, f1 = eval_func(import_name=import_name,
                            mode=mode,
                            dataset=data,
                            repetition=REPETITIONS,
                            )
                
                with open(report_path + f'{REPORT_NAME}.txt', 'a') as f:
                    f.write(30*'--' + '\n')
                    f.write(f'data: {data}, pretrain: {pretrain}, cap: {cap*100:.0f}% --> IoU: {iou[2]:.3f}; F1:{f1[2]:.3f}\n')
                    f.write(f'{iou[0]:.3f} {iou[1]:.3f} {iou[2]:.3f} ...... {f1[0]:.3f} {f1[1]:.3f} {f1[2]:.3f}\n')


if __name__ == "__main__":
    main()