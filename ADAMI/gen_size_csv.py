import numpy as np
from PIL import Image
from os.path import join

img_pair_list = list()
with open(r'', 'r') as f:
    img_in_csv = f.read().split('\n')[1:-1]
    img_pair_list += img_in_csv
with open(join(r'E:\Paper\PythonPro\SFUDA\SFDA-AdaMI-main\sizes', 'MESSIDOR_Base3_unlabeled.csv'), 'w') as f:
    f.write('val_ids,dumbpredwtags\n')
    for i in img_pair_list:
        f.write('{},"{}"\n'.format(i, "[15786, 3394]"))

# # MESSIDOR_Base1  "[21746, 6794]"
# tif_image = r'E:\Dataset\RIGAPlus\RIGA-mask\MESSIDOR_Base1\Labeled\image1-1.tif'

# # MESSIDOR_Base2  "[16814, 5280]"
# tif_image = r'E:\Dataset\RIGAPlus\RIGA-mask\MESSIDOR_Base2\Labeled\image174-1.tif'

# # MESSIDOR_Base3  "[15786, 3394]"
# tif_image = r'E:\Dataset\RIGAPlus\RIGA-mask\MESSIDOR_Base3\Labeled\image322-1.tif'

# # REFUGE         "[36331, 15125]"
# tif_image = r'E:\Dataset\RIGAPlus\RIGA-mask\REFUGE_1\Labeled\g0001-1.tif'

# # DS              "[67801, 44365]"
# tif_image = r'E:\Dataset\RIGAPlus\RIGA-mask\DS\Labeled\drishtiGS_001-1.tif'


# label = Image.open(tif_image)
#
# label = label.resize((512, 512), resample=Image.NEAREST)
#
# label_npy = np.array(label)
# od = np.zeros_like(label_npy)
# oc = np.zeros_like(label_npy)
# od[label_npy > 0] = 1
# oc[label_npy == 128] = 1
#
# sum_od = np.sum(od)
# sum_oc = np.sum(oc)
# print(str(sum_od)+', '+str(sum_oc))
