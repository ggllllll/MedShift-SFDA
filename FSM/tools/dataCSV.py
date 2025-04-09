import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from os.path import join

train_data_type = '.tif'
mask_data_type = '.tif'

Original_vali_img_path = r''
Original_vali_img_path_1 = r''

file_train_list = [file for file in os.listdir(Original_vali_img_path) if file.lower().endswith(train_data_type)]
print(str(len(file_train_list)))

# # FSM 无标记的
# with open(join(Original_vali_img_path_1, 'FSM_fine.csv'), 'w') as f:
#     f.write('image,fine,pseudo\n')
#     for i in file_train_list:
#         f.write('RIGA/RIM-ONE-r3/Unlabeled/{},RIGA-FSM/RIM-ONE-r3/Unlabeled/fine/{},RIGA-FSM/RIM-ONE-r3/Unlabeled/pseudo/{}\n'.format(i, i, i.replace('.tif', '-1.tif')))

# # FSM 有标记的
with open(join(Original_vali_img_path_1, 'FSM_fine.csv'), 'w') as f:
    f.write('fine,mask\n')
    for i in file_train_list:
        f.write('RIGA-FSM/RIM-ONE-r3/Labeled/fine/{},RIGA-mask/RIM-ONE-r3/Labeled/{}\n'.format(i, i.replace('.tif', '-1.tif')))
