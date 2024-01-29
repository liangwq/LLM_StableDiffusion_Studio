---
title: LLM StableDiffusion Studio WebUI
---
## 理论文章 ##
https://blog.csdn.net/liangwqi/article/details/130300586?spm=1001.2014.3001.5501

https://zhuanlan.zhihu.com/p/623924586

## 整合LLM大模型的StableDiffussion工作空间 ##
Pip install -r requirement.txt

python studio_root.py 进入可视化界面

## clip retrieval理论部分 ##
1.**[知乎链接](https://zhuanlan.zhihu.com/p/680405647)**

代码见APP_example/clip_retrieval
<div>
<code>
1.图片库特征抽取代码：extract_embeddings.py
2.图片特征在faiss向量数据库建立索引：build_index.py
3.可视化应用界面：app.py
</code>
</div>

![clip_search00](https://github.com/liangwq/Chatglm_lora_multi-gpu/assets/9170648/93d5c672-39da-44cd-9ed3-12ac0c5a50c8)

![clip_searcg01](https://github.com/liangwq/Chatglm_lora_multi-gpu/assets/9170648/32f07e1b-70d3-4cfe-837d-707c7dac6195)

## 致力打造中国版StableDiffusion WebUI ##
本项目持续维护，会集成更多工具，让工具本身更智能顺手
对本项目有兴趣的同学可以一起参与代码维护

![F4A3E0B1-C9B9-4DA8-A862-1D3754E41DDA](https://user-images.githubusercontent.com/9170648/234258106-3ed0aca0-4ddb-4df8-a8e5-b7b1ca3408cc.png)

<img width="1259" alt="截屏2023-04-18 下午7 48 50" src="https://user-images.githubusercontent.com/9170648/233757706-91e09429-7a22-403d-af48-8ba3a4c66c58.png">

<img width="1272" alt="截屏2023-04-18 下午7 49 58" src="https://user-images.githubusercontent.com/9170648/233757788-77491d86-79df-4427-9dee-f180fdb97d79.png">

<img width="1265" alt="截屏2023-04-18 下午7 52 07" src="https://user-images.githubusercontent.com/9170648/233757805-e079f7b2-09ab-4225-a2be-fccb82297e58.png">

<img width="1257" alt="截屏2023-04-18 上午8 43 06" src="https://user-images.githubusercontent.com/9170648/233757916-41388d9f-8470-4ccb-8f8e-ed013ea9f221.png">
