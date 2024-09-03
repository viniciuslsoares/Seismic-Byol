from evaluate import eval_func
import numpy as np


"""
Com essa função é possível avaliar modelos em sequência dentro de um loop.
A ideia é usar listas e dicionários para armazenar os parâmetros de cada modelo.    
"""


def main():
    
    REPORT_NAME = 'eval_parihaka_13[0]'

    report_path = 'reports/'
            
    # list_of_datas = ['f3', 'seam_ai']
    # list_of_datas = ['f3']
    list_of_datas = ['seam_ai']
    
    # list_of_pretrains = ['f3', 'seam_ai', 'COCO', 'IMAGENET', 'both', 'sup']
    list_of_pretrains = ['f3', 'seam_ai', 'both']
    # list_of_pretrains = ['COCO', 'IMAGENET', 'sup']
    # list_of_pretrains = ['seam_ai']

    # teste
        
    list_of_repets = ['V01', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10']
    # list_of_repets = ['V01', 'V2', 'V3']
        
    list_of_caps = [0.01, 0.1, 0.5, 1.0]
    
    with open(report_path + f'{REPORT_NAME}.txt', 'w') as f:
        f.write('Report of the evaluation of the models\n')
        f.write('---------------------------------------\n')
        f.write(f'Datas being used: {list_of_datas}\n')
        f.write(f'Pretrains models: {list_of_pretrains}\n')
        f.write(f'Caps: {list_of_caps}\n')
        
    
    
    
    for data in list_of_datas:
        for pretrain in list_of_pretrains:
            
            with open(report_path + f'{REPORT_NAME}.txt', 'a') as f:
                f.write(f'------------------ Pre on {pretrain} and train on {data} ------------------\n')
            
            for cap in list_of_caps:

                print(30*'*-')
                print(f'''Running with data {data} and model pretrained in  {pretrain} with cap {cap*100:.0f}%. Evaluating test dataset''')
                print(30*'*-')
                list_of_iou = []
                list_of_f1 = []    
                
                for repetition in list_of_repets:
                    

                    if pretrain == 'f3' or pretrain == 'seam_ai' or pretrain == 'both':
                        mode = 'byol'
                        import_name = f'{repetition}_pre_{pretrain}_train_{data}_cap_{cap*100:.0f}%'
                    elif pretrain == 'COCO':
                        mode = 'coco'
                        import_name = f'{repetition}_pre_COCO_train_{data}_cap_{cap*100:.0f}%'
                    elif pretrain == 'IMAGENET':
                        mode = 'imagenet'
                        import_name = f'{repetition}_pre_IMAGENET_train_{data}_cap_{cap*100:.0f}%'
                    elif pretrain == 'sup':
                        mode = 'supervised'
                        import_name = f'{repetition}_sup_{data}_cap_{cap*100:.0f}%'


                    iou, f1 = eval_func(import_name=import_name,
                                mode=mode,
                                dataset=data,
                                repetition=repetition,
                                )

                    list_of_iou.append(iou[2])
                    list_of_f1.append(f1[2])
                    
                    with open(report_path + f'{REPORT_NAME}.txt', 'a') as f:
                        f.write(30*'--' + '\n')
                        f.write(import_name + '\n')
                        f.write(f'data: {data}, pretrain: {pretrain}, cap: {cap*100:.0f}% --> IoU: {iou[2]:.3f}; F1:{f1[2]:.3f}\n')
                    
                with open(report_path + f'{REPORT_NAME}.txt', 'a') as f:
                    # f.write(30*'--' + '\n')
                    # f.write(import_name + '\n')
                    # f.write(f'data: {data}, pretrain: {pretrain}, cap: {cap*100:.0f}% --> IoU: {iou[2]:.3f}; F1:{f1[2]:.3f}\n')
                    # f.write(f'{iou[0]:.3f} {iou[1]:.3f} {iou[2]:.3f} ...... {f1[0]:.3f} {f1[1]:.3f} {f1[2]:.3f}\n')
                    f.write(f'mean: {np.mean(list_of_iou):.2f}; std: {np.std(list_of_iou):.2f}\n')
                    f.write(f'max: {np.max(list_of_iou):.2f}; min: {np.min(list_of_iou):.2f}\n')
                    # f.write(f'f1: {list_of_f1}\n')


if __name__ == "__main__":
    main()