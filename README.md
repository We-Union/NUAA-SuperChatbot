# NUAA-SuperChatbot
NUAA的自定义机器学习课设，实现了一个语音级端到端的聊天机器人

项目的文件夹的结构为：

- ASR : 语音识别模块，负责将用户输入的语音转换为文本序列
- ChatBot : 聊天机器人模块，负责将输入的中文文本序列转换为对应输出的语音序列
- TTS : 文语转换模块，也就是语音合成模块

--
-
## ChatBot

### 介绍
属于整个系统的核心组件，可以理解为将一句中文“翻译”成中文，从而在功能上达到对话的效果。

### 说明
- 请不要在`./ChatBot`文件夹下运行程序，因为该文件夹已经被声明为一个**Python模块**
- 所有的训练参数，模型参数，推理参数，其他参数都写在了`./ChatBot/constants.py`中，你可以打开这个文件夹或者在运行时修改其中的参数。
- 为了方便调整参数，我已经将一组迷你样本放在了`./data/ChatBot/ensemble`中，包括迷你对话索引序列数据集和一个词表数据集。


### 开箱调参
如果你需要为模型设置参数，那么请打开单元测试文件`./unit_test.py`，运行测试训练的单元函数`test_train()`即可。这个函数中的训练函数说明如下：

```python
train(
        version="0.0.0",                    # 当前的版本，这是为了更好的进行版本控制
        pairs=data_dict["index_pairs"],     
        Epoch=EPOCH_NUM,                    # 训练论述，默认采用模块默认的，这个参数可以在constant.py中找到
        model=model,                        
        optimizer=optimizer,    
        batch_size=BATCH_SIZE,              # Batch的大小
        save_dir="./dist/ChatBot",          # 保存模型文件的路径文件夹
        save_model=True,                    # 是否保存模型
        save_optimizer=False,               # 是否保存优化器参数
        clip_threshold=CLIP_THRESHOLD,      # 梯度裁剪参数
        TF_ratio=TEACHER_FORCING_RATE,      # 使用TF策略的概率
        save_interval=SAVE_INTERVAL,        # 保存模型的间隔，默认每隔SAVE_INTERVAL轮保存一次模型
        display_progress_bar=True           # 是否开启进度条，强烈建议开启，因为被我调教得很好看
    )
```

你可以在`./ChatBot/constants.py`中修改参数以适配训练，当打印信息让你比较满意时，你就可以考虑开始在超大样本上开始训练了。

## 使用
1. 首先使用`pip install -r requirements.txt`安装依赖。
2. 使用`python manage.py runserver`启动项目，默认只有本机可访问，且端口为8000。