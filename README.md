# ugrip
Baby Cry Update Code

current plans:
- Pad audio
- Augment data using DCASE23 & UrbanSounds8K
- implement evaluation metrics
- implement temporal contrastive loss function

Ben's code
- transformer architecture that runs very quickly and has an accuracy around 86%
- implements Danning's metrics function so we can get a more accurate accuracy metric (generally around 80%)
- trains on Ben & Saad's annotations, as well as 100 DCASE non-cry audio sets

Danning's code:
- metrics

BabyCry data uploaded:
- The names are trimmed to the first 47 characters to avoid conflict.
- This is because Textgrid names converted _ to - at the end of the file, so we trimmed to keep relevant identification
