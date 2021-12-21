## SIA-GAN
#### Signal and Image Attention GAN

## 개요
#### ECG Signal과 STFT를 통해 이미지화 된 ECG를 이용한 GAN 기반의 multimodal 심부전 탐지 알고리즘

## 목표
#### signal과 freq의 feature를 transformer를 활용해 효과적으로 fusion하는 GAN기반의 심부전 탐지 알고리즘 개발

## 파일 구조
 ### 1. experiments
 - 1학기동안 진행한 다양한 실험들
 - 실험 결과표
 ![image](https://user-images.githubusercontent.com/50744156/146861028-14b54205-02d4-46ee-8a0e-e13e106b0746.png)


 ### 2. final_model
 - 실험 후에 얻은 SIA-GAN의 최종 모델
 - #### 모델구조
  ![image](https://user-images.githubusercontent.com/50744156/146860866-c1ff6a99-43c5-4c5c-b746-53d93aa37062.png)
 - 최종 AUC : 0.9588
 - #### signal 생성결과
    ![image](https://user-images.githubusercontent.com/50744156/146861442-bcdf63e6-a2a7-41c4-a1da-3a07af11390a.png)
     - 정상 ECG를 입력하면 정상 ECG에 가까운 fake signal을 생성하기 때문에 fake signal과 input signal의 차이가 거의 없음, 
     - 비정상 ECG를 입력하면 generator에서 정상 ECG에 가까운 fake signal을 생성, input signal과 fake signal의 차이가 나는 부분으로 비정상 탐지 가능
 
 ### 3. docs
   - SIA-GAN 실험에 대한 더 자세한 정보를 담은 참고자료

## 타 모델과의 비교
![image](https://user-images.githubusercontent.com/50744156/146861925-2c75bdae-d217-415c-8e59-1d9dd868ef0c.png)


