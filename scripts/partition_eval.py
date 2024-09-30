from evaluate import eval_func
import numpy as np





def main():
    
    report_name = 'folds_eval_3'
    
    data = 'f3'
    main_path = '../../shared_data/seismic/f3_segmentation_'
    # list_of_partitions = ['f3_fold_0', 'f3_fold_1', 'f3_fold_2', 'f3_fold_3', 'f3_fold_4']
    # list_of_models = ['fold_0', 'fold_1', 'fold_2', 'fold_3', 'fold_4']
    repetition = 'V_0.01'
    list_of_partitions = ['dataset_random/', 'dataset_uniform/', 'dataset_window/', 'highest_mse_pair/']
    list_of_models = ['random', 'uniform', 'window', 'high_mse_pair']
    
    list_of_models = ['window_1', 'mse_pair_1']
    list_of_partitions = ['dataset_window/', 'highest_mse_pair/']
    
    with open(report_name + '.txt', 'w') as f:
        f.write('---------------------------------------\n')
        f.write(f'Partitions: {list_of_partitions}\n')
        f.write(f'Models: {list_of_models}\n')
        f.write('---------------------------------------\n')
    
    for i in range(len(list_of_partitions)):
        partition = list_of_partitions[i]
        model = list_of_models[i]
        
        root_dir = main_path + partition
        
        with open(report_name + '.txt', 'a') as f:
            f.write(f'------------------ {partition} ------------------\n')
        
        iou, f1 = eval_func(import_name=model,
                  mode='supervised',
                  dataset=data,
                  repetition=repetition,
                  root_dir=root_dir,
                  )

        with open(report_name + '.txt', 'a') as f:
            f.write(30*'--' + '\n')
            f.write(model + '\n')
            f.write(f'iou = {iou}\n')
            f.write(f'f1 = {f1}\n')            
    
    
    



if __name__ == "__main__":
    main()