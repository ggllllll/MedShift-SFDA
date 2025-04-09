# -*- coding:utf-8 -*-
def convert_labeled_list_lab(csv_list):
    img_pair_list = list()
    for csv_file in csv_list:
        with open(csv_file, 'r') as f:
            img_in_csv = f.read().split('\n')[1:-1]
        img_pair_list += img_in_csv
    fine_list = [i.split(',')[0] for i in img_pair_list]
    mask_list= [i.split(',')[1] for i in img_pair_list]
    return fine_list, mask_list
def convert_labeled_list_Unlab(csv_list):
    img_pair_list = list()
    for csv_file in csv_list:
        with open(csv_file, 'r') as f:
            img_in_csv = f.read().split('\n')[1:-1]
        img_pair_list += img_in_csv
    image_list = [i.split(',')[0] for i in img_pair_list]
    fine_list= [i.split(',')[1] for i in img_pair_list]
    pseudo_list= [i.split(',')[2] for i in img_pair_list]
    return image_list, fine_list, pseudo_list

# TEST_TARGET_LIST = [r'E:\Dataset\RIGAPlus\RIGA-FSM\MESSIDOR_Base1\Unlabeled\FSM_fine.csv']
# img_list_1, label_list_2, label_list_3 = convert_labeled_list_Unlab(TEST_TARGET_LIST)