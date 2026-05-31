# Starting from scripts/train.py (line 21), what is the full call chain that leads to one UCOD-DPL training step in engine/runner/loop_UCOD_DPL.py (line 143)?
The full call chain is:
```text
scripts/train.py
main()
-> parse_train_args()
-> init_cfg(args)
-> StandardRunner(cfg)
-> BaseRunner.__init__()
-> BaseRunner._initialize_components()
-> StandardRunner._build_model()
-> StandardRunner._build_optimizer()
-> StandardRunner._build_dataloader()
-> StandardRunner._prepare_accelerator()
-> runner.launch_train()
-> TrainLoop(self.config, self)
-> TrainLoop.run()
-> TrainLoop.run_epoch()
-> TrainLoop._process_batch(batch_data)
```
# The paper says training is unsupervised and uses fixed-strategy pseudo-labels. Where are these pseudo-labels generated, cached, and loaded? Check generate_pseudo_label.py (line 57), data/utils/found_bkg_mask.py (line 4), data/datasets/base_dataset.py (line 123), and data/datasets/cache_manager.py (line 47).
```text
Generated:
generate_pseudo_label.py
main()
-> generate_mask(image_path, th_bkg=0.6)
-> compute_img_bkg_seg()
-> invert background mask with (1 - mask)
-> refine_post_process()
-> append result to mask_list

Cached:
generate_pseudo_label.py
main()
-> MetaListPickleIO(base_path=cache_path/dataset)
-> cacheio.dump_list(mask_list)
-> MetaListPickleIO.dump_list()
-> MetaListPickleIO.write_file()
-> PickleIO.write_file()
-> index.json records index-to-pickle mapping

Loaded:
BaseCODDataset.__getitem__()
-> self.cache_manager.get_pseudo_label_cache()
-> MultiCacheManager.get_cache("pseudo_label")
-> CacheManager.read_file(index)
-> MetaListPickleIO.read_file(index)
-> PickleIO.read_file()
```
# The paper’s teacher-student framework says the teacher is updated by EMA with momentum 0.99. Where is the teacher represented in code, where is EMA updated, and which config sets 0.99? Start with models/uscod.py (line 8), engine/runner/loop_UCOD_DPL.py (line 176), and configs/uscod/UCOD-DPL_dinov2.py (line 41).

```text
teacher:
uscod.py
baseline().decoder_ema
forward(..., ema=True) uses decoder_ema, so this is the teacher / EMA branch.

student:
models/uscod.py
baseline().decoder
forward(..., ema=False) uses decoder, so this is the student branch.

EMA is updated:
loop_UCOD_DPL.py
TrainLoop()._process_batch()
-> optimizer.step()
-> lr_scheduler.step()
-> update_ema_decoder()

EMA update rule:
update_ema_decoder()
-> decoder_ema parameters are updated from decoder parameters
-> alpha = min(1 - 1 / (global_step + 1), ema_weight)

config sets 0.99:
configs/uscod/UCOD-DPL_dinov2.py
model_cfg.ema_weight = 0.99
```

# The paper’s APM mixes fixed pseudo-labels with teacher predictions using discriminator scores. How does merge_pseudo_label (line 255) implement this idea, and what is similar or different from the paper’s Eq. 3?
```text
merge_pseudo_label() implements APM by comparing discriminator scores for:
1. the student prediction p_students
2. the fixed pseudo-label pseudo_labels

First, it binarizes teacher and student predictions:
p_teachers = (p_teachers.sigmoid() > 0.5).float()
p_students = (p_students.sigmoid() > 0.5).float()

Then it computes discriminator scores:
p_s = discriminator(p_students, features)
p_p = discriminator((pseudo_labels > 0.5).float(), features)

It computes a mixing weight:
weight = 0.5 * (1 + cos(abs(p_s - p_p) * pi))
weight += current_epoch / (max_epoch + start_finetune)
weight = clamp(weight, 0, 1)

Finally, it mixes fixed pseudo-labels and teacher predictions:
mixed_label = pseudo_labels * (1 - weight) + p_teachers * weight
``` 
# The paper says the discriminator is trained alternately with the model. Where does the code freeze/unfreeze the discriminator and decoder, and what labels are assigned to pseudo-label masks versus student predictions? Inspect Discriminator_train (line 217), Discriminator_epoch (line 229), and models/discriminator.py (line 60).
 ```text
Before training the discriminator:
- discriminator is unfreezed
- student decoder is freezed

Discriminator_train()
for param in self.runner.discriminator.parameters():
	param.requires_grad  = True
for param in self.runner.model.decoder.parameters():
	param.requires_grad = False

During discriminator training:
Discriminator_epoch()
- student predictions are generated under torch.no_grad()
- pseudo-label masks are binarized
- discriminator scores both masks

labels:
0 = student predictions
1 = fixed pseudo-label masks

probs = torch.cat((probs_student, probs_pseudo), dim=0)
label = torch.cat((zeros(batch_size), ones(batch_size)), dim=-1).unsqueeze(-1)

After discriminator training:
- discriminator is frozen again
- student decoder is unfrozen again

for param in self.runner.discriminator.parameters():
	param.requires_grad  = False
for name, param in self.runner.model.decoder.named_parameters():
	if 'decoder_ema' not in name: # redundant
		param.requires_grad = True 
		
labels
 0 is assigned to student predictions
 1 is assigned to pseudo-label masks

probs = torch.cat((probs_student, probs_pseudo), dim=0)
label = torch.cat((torch.zeros(batch_size),torch.ones(batch_size)), dim=-1).unsqueeze(-1).to('cuda')
 ```
# The paper’s DBA decoder has foreground and background branches plus an orthogonal loss. How does models/modules/DBA.py (line 4) implement the two branches, and where are foreground loss, background loss, and orthogonal loss added during training?
```
DBA two branches:
models/modules/DBA.py

RevDecoder first projects the input feature:
decoupling = Conv2d(feature_dim, 2 * embedding_dim)

Then it splits the projected feature into two branches:
dfeatures_1, dfeatures_2 = torch.chunk(decoupled_feat, 2, dim=1)

Branch 1:
features_1 -> attention_1 -> conv_out_fg -> fg_mask

Branch 2:
features_2 -> attention_2 -> conv_out_bg -> bg_mask

Orthogonal loss:
extra_loss = calc_orthogonal_loss(features_1, features_2)
return fg_mask, bg_mask, extra_loss

Training loss:
engine/runner/loop_UCOD_DPL.py

preds, preds_rev, extra_loss = self.runner.model(features)

foreground BCE:
loss = BCE(preds, mixed_pseudo_labels)

background BCE:
loss += BCE(preds_rev, 1 - mixed_pseudo_labels)

orthogonal loss:
if extra_loss is not None:
    loss += extra_loss

APM/adversarial term:
pseudo_labels, dis_loss = merge_pseudo_label(...)
if dis_loss is not None and not self.finetune:
    loss -= dis_loss
```

# The paper’s Look-Twice mechanism refines small objects and uses threshold 0.15 for DINOv2. Where is that threshold configured, and how does the validation loop decide whether to crop/re-infer? Check configs/uscod/UCOD-DPL_dinov2.py (line 31) and ValLoop_Look_Twice (line 274).
```text
Threshold:
configs/uscod/UCOD-DPL_dinov2.py
val_cfg.look_twice = True
val_cfg.look_twice_th = 0.15
val_cfg.expand_type = "dynamic"

Crop / re-infer decision:
engine/runner/loop_UCOD_DPL.py
ValLoop_Look_Twice.run()
-> model predicts coarse mask
-> process_preds(preds, labels)

process_preds():
-> upsample prediction to validation image size
-> apply sigmoid and threshold > 0.5
-> connectedComponents() finds foreground components
-> p = component_area / full_image_area
-> p_max = largest component ratio

If:
p_max < self.cfg.val_cfg.look_twice_th

Then:
-> collect bounding boxes for components with p[i] > 0.01
-> expand each bbox using expand_bbox()
-> return bboxes

Re-infer:
ValLoop_Look_Twice.run()
if bboxes is not None and look_twice is enabled:
    preds_up = look_twice(img_path, bboxes, preds_up)

look_twice():
-> crop each bbox from original image
-> resize / normalize crop
-> extract features
-> run model again on crop
-> resize crop prediction back to bbox size
-> paste refined prediction into the original coarse mask
```    
# The paper evaluates CHAMELEON, CAMO, COD10K, and NC4K with S-measure, E-measure, F-measure, weighted F-measure, and MAE. Where does the repo select these datasets, and where are the metrics computed/reported? Start with scripts/eval.py (line 8) and engine/utils/metrics/metric.py (line 14).
```text
Select datasets
scripts/eval.py
DATASET = ['CHAMELEON', 'TE-CAMO', 'TE-COD10K', 'NC4K']

main()
-> for dataset in DATASET:
-> cfg.dataset_cfg.valset_cfg.DATASET = dataset
-> runner = Runner(cfg)
-> runner.launch_val_look_twice()

Compute/report the metrics:
engine/runner/loop_UCOD_DPL.py
ValLoop_Look_Twice.run()
-> statistics_val = statistics()
-> statistics_val.step(all_labels, preds_up > 0.5)
-> result = statistics_val.get_result()
-> result_table = {key: [round(result[key], 4)] for key in result.keys()}
-> logger.log_table(result_table)

Metric implementation:
engine/utils/metrics/metric.py

statistics.__init__()
-> MAEmeasure()
-> Smeasure()
-> Emeasure()
-> Fmeasure()
-> WeightedFmeasure()
-> ACCmeasure()
-> IOUmeasure()

statistics.get_result()
returns:
ACC, mIOU, E_MAX, E_MEAN, F_MAX, F_MEAN, SMeasure, MAE, WFM
```