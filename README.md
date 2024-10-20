**1. Diffusion**
   - 지문과 동일 방식으로 Diffusion network 학습(fingerprint/diffusion 폴더)
+ 원본 데이터인 VERA-FV-full-bf 에 위아래 padding을 추가하여 학습
Generate_sample_diffusion.py 실행(생성시 ckpt 경로 입력)

![image](https://github.com/user-attachments/assets/81666198-9458-4e03-9d70-2d05d14d8594)

**2. Ridgepattern-GAN(thumbnail)**
  
**3. Feature extraction**
![image](https://github.com/user-attachments/assets/95ee62ce-09de-4bcc-a4be-65df43c863a7)

**4. ID-preserving network**
   Data_processing 폴더의 extract_skeleton.py 사용
   python train.py --modality fingervein --net_name D_IDPreserve --data_dir ../dataset/Fingervein/fingervein_pair_padding_train  --exp_name fingervein_D_IDPreserve_padding --save_epochs 20 --epochs 1600 --gpu_ids=0,1,2,3
![image](https://github.com/user-attachments/assets/12bb4f38-d400-4947-8d66-d62851e5d610)
