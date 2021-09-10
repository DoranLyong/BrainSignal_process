# BrainSignal_process
집중도인식 DB EDA



### 1. count_main.py

* 행동별 뇌파의 평균수치를 계산하는 코드 

```bash
indiv_Means.csv
indiv_Nums.csv
global_Means.csv
global_Nums.csv
```



### 2. re-anno_main.py

* 각 행동 카테고리별 binary vector 형태로된 레이블링을 decimal로 변환 
* 각 피험자별 전체 행동 카테고리를 통합해서 생성 
* `anno` → `re-anno`  로 개선됨 



### 3. make_meta_csv.py

* [dynamic-images-for-action-recognition](https://github.com/DoranLyong/dynamic-images-for-action-recognition/tree/master/dynamic_image_networks/hmdb51/preprocessing) 에서 사용하는 metadata_split.csv 와 동일한 포멧 생성 





