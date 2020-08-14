#  基于distillBERT多分类任务

基于[微信文章](https://mp.weixin.qq.com/s/ogQ8_0Wr3855uNEo328Vew)做的多分类任务

##  数据集

链接https://www.kesci.com/home/dataset/5e7a0d6398d4a8002d2cd201/files

##  环境要求

```
simpletransformers==0.47.0
pandas==0.25.3
smart-open==2.1.0
numpy==1.18.5
scikit-learn==0.22.1
tensorflow==1.12.0
torch==1.2.0
```

##  simpletransformers

simpletransformers库是方便使用transformer模型的库。

**Classification Models**  

```python
simpletransformers.classification.ClassificationModel(self, model_type, model_name, num_labels=None, weight=None, args=None, use_cuda=True, cuda_device=-1, **kwargs,)
```

*Parameters*

- **model_type** *(`str`)* - The type of model to use ([model types](https://simpletransformers.ai/docs/classification-specifics/#supported-model-types))
- **model_name** *(`str`)* - The exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.
- **num_labels** *(`int`, optional)* - The number of labels or classes in the dataset. (See [here](https://simpletransformers.ai/docs/classification-models/#specifying-the-number-of-classeslabels))
- **args** *(`dict`, optional)* - [Default args](https://simpletransformers.ai/docs/usage/#configuring-a-simple-transformers-model) will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args.

**Training a Classification Model**

```
simpletransformers.classification.ClassificationModel.train_model(self, train_df, multi_label=False, output_dir=None, show_running_loss=True, args=None, eval_df=None, verbose=True, **kwargs)
```

*Parameters*

- **train_df** - Pandas DataFrame containing the train data. Refer to [Data Format](https://simpletransformers.ai/docs/classification-data-formats/).
- **args** *(`dict`, optional)* - A dict of configuration options for the `ClassificationModel`. Any changes made will persist for the model.

**参考文章**

- https://simpletransformers.ai/docs/classification-specifics/
- https://simpletransformers.ai/docs/classification-models/#training-a-classification-model

##  自己遇到的一些问题

标签与文本分开时，遇到问题。

原代码

```python
split_data1 = [i.split('==@++') for i  in data1]
labels1 = [i[0] for i in split_data1]
text1 = [i[1] for i in split_data1]

split_data2 = [i.split(r'\t') for i  in data2]
labels2 = [i[0] for i in split_data2]
text2 = [i[1] for i in split_data2]
```

改之后

```python
sp_data1 = [i.split('==@++') for i in data1]
split_data1 = [i[0].split(r'\t') for i in sp_data1]
labels1 = [i[0] for i in split_data1] 
text1 = [i[1] for i in split_data1]
sp_data2 = [i.split('==@++') for i in data2]
split_data2 = [i[0].split(r'\t') for i in sp_data2]
labels2 = [i[0] for i in split_data2]
text2 = [i[1] for i in split_data2]
```

原

---------------

在运行到`model.train_model()`时遇到的问题。

```linux
Can't pickle local object 'get_linear_schedule_with_warmup.<locals>.lr_lambd
```

解决方法：当时pytorch=0.4，将pytorch升级到1.3便没有这个问题。

--------

在老机器上遇到的问题。

```
import torch
“非法指令（核心已转储）”
```

解决办法：降低pytorch版本