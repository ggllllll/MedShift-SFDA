CC = python
SHELL = bash
PP = PYTHONPATH="$(PYTHONPATH):."

.PHONY: all view plot report

CFLAGS = -O
#DEBUG = --debug

#the regex of the slices in the target dataset for the ivd
G_RGX = Subj_\d+_\d+

TT_DATA = [('Inn', png_transform, False), ('GT', gtpng_transform, False),('GT', gtpng_transform, False)]
S_DATA = [('Wat', png_transform, False), ('GT', gtpng_transform, False),('GT', gtpng_transform, False)]
L_OR = [('CrossEntropy', {'idc': [0,1], 'weights':[1,1]}, None, None, None, 1)]
NET = UNet

#the network weights used as initialization of the adaptation
M_WEIGHTS_ul = results/ivd/cesource/last.pkl

#run the main experiments
TRN = results/ivd/cesource results/ivd/sfda

# first train on the source dataset only:
results/ivd/cesource: OPT =  --target_losses="$(L_OR)" --target_dataset "data/ivd_transverse/" --target_folders="$(S_DATA)" --val_target_folders="$(S_DATA)" \
	     --network UNet --model_weights="" --lr_decay 1 \


# SFDA. Put --saveim False and remove --entmap and --do_hd 90 to speed up
results/ivd/sfda: OPT = --target_losses="[('EntKLProp', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sizefile':'sizes/prostate.csv'},'norm_soft_size',1)]" \
          --do_hd 90 --saveim True --entmap --l_rate 0.000001 --n_epoch 100 --lr_decay 0.9 --model_weights="$(M_WEIGHTS_ul)" \

#inference mode : saves the segmentation masks for a specific model saved as pkl file (ex. "results/sa/cesource/last.pkl" below):
results/ivd/cesourceim: OPT =  --target_losses="$(L_OR)" \
	   --mode makeim  --batch_size 1  --l_rate 0 --model_weights="results/sa/cesource/last.pkl" --pprint --lr_decay 1 --n_epoch 1 --saveim True\

$(TRN) :
	$(warning $(OPT))




