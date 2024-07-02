# MBZUAI Ugrip24 project

![image](https://github.com/SaadElDine/Baby-Cry-Detection-UGRIP/assets/113860522/5c7a33fd-d180-4c8c-8572-2391cc9f6547)

# Cappellas's current model problems:
- Background noise
- Confusion Between Baby's Cry and Baby's Laughter
- Adults crying

# Current model Demo:
  

# Our Work:
  ![image](https://github.com/SaadElDine/Baby-Cry-Detection-UGRIP/assets/113860522/97949ffd-0e17-4707-b7c8-82c0a3215685)

1. Data Annotation using praat (Dataset: Donate a cry Kaggle)
  ![image](https://github.com/SaadElDine/Baby-Cry-Detection-UGRIP/assets/113860522/b3ab1f3e-fe5b-4902-9367-72652973bef4)

  ![image](https://github.com/SaadElDine/Baby-Cry-Detection-UGRIP/assets/113860522/fbd6a584-3a31-4e9f-bd99-4a24aec4c2f8)

2. Audio Preproccessing and Features Extraction
   ![image](https://github.com/SaadElDine/Baby-Cry-Detection-UGRIP/assets/113860522/56286be5-3465-4742-a470-85cd692de037)

   - Try Mel-Spectogram And MFCC an we Used Mel-Spectogram
  ![image](https://github.com/SaadElDine/Baby-Cry-Detection-UGRIP/assets/113860522/c148f64c-e727-4895-9da9-72ca53595daf)
     
 
3. Model Architecture Tried
   - CNN
   - CNN-LSTM
   - MobileNet-V2
   - Bi-LSTM
   - Transformer With Multihead Attention
   - Audio Spectogram Transformer AST (Using LiFormer = Linear Projection)
   - Transformer With Flash Attention
   - TinyFormer
   - MarvelNet

4. Model Optimization
     ![image](https://github.com/SaadElDine/Baby-Cry-Detection-UGRIP/assets/113860522/a471b480-5b76-4b3f-aef4-09104ef72ba9)

5. Models Evaluation
     ![image](https://github.com/SaadElDine/Baby-Cry-Detection-UGRIP/assets/113860522/ca34869f-30d3-4b9c-9630-f5452908d7ab)
   - Explanation:
     ![image](https://github.com/SaadElDine/Baby-Cry-Detection-UGRIP/assets/113860522/10df1d4e-63a3-44b2-a3ac-fb23729dcafb)
     
6. Comparing Performancxe
   ![image](https://github.com/SaadElDine/Baby-Cry-Detection-UGRIP/assets/113860522/11181d08-c842-489d-82a7-624fa766186e)
 
   ![image](https://github.com/SaadElDine/Baby-Cry-Detection-UGRIP/assets/113860522/9f34a031-5ab7-4271-9c4a-8caf181387c7)

7. Choose Best Model (MobileNet-V2)
   ![image](https://github.com/SaadElDine/Baby-Cry-Detection-UGRIP/assets/113860522/2aa4819a-f783-4f39-9e9d-3fe438de718b)

8. Demo for Testing the model using diffrenert audios
   https://github.com/SaadElDine/Baby-Cry-Detection-UGRIP/assets/113860522/9c2702fc-9a0e-40f2-b9e0-69a48183f697


# What is next for publication?
- Data Augmentation
- Include Video of babies
- Model optimization




