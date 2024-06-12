# ugrip
Baby Cry Update Code

current plans:
- Pad audio
- Augment data using DCASE23 & UrbanSounds8K
- implement evaluation metrics
- implement temporal contrastive loss function

Ben's code
- transformer architecture that runs very quickly and has an accuracy around 86%
- lots of bugs and messy code currently, will be doing more advanced metrics soon

Danning's code:
- metrics

Mohamed's code:
- Custom dataset for Babycry. Works regardless of target number of samples, target sample rate, audio sample rate.
  
BabyCry data uploaded:
- The names are trimmed to the first 47 characters to avoid conflict.
- This is because Textgrid names converted _ to - at the end of the file, so we trimmed to keep relevant identification
