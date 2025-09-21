# YOLO11 在 RKNN 开发板上的部署指南

为了方便在 RKNN 开发板上部署 YOLO11 模型，整理了完整流程，避免配置错误和遗忘关键步骤。

---

## 1. 配置 YOLO11 环境

在笔记本或开发板上首先配置 YOLO11 环境（PyTorch + Ultralytics）。

⚠️ 注意：RKNN 官方修改过 Ultralytics，导出的 ONNX 模型与官方原版不同。  
可使用 **Netron** 查看 ONNX 文件差异。

推荐流程：

1. 正常安装 YOLO11 环境（建议使用 Anaconda）。
2. 卸载环境中的官方 `ultralytics`：
```
pip uninstall ultralytics
```
3. 下载本仓库的 `ultralytics_yolo11` 到本地，在该工作空间中进行 ONNX 转换。

---

## 2. 导出 ONNX 模型

确保环境满足 `./requirements.txt` 后执行：

```
# 修改 ./ultralytics/cfg/default.yaml 中 model 路径：
# - yolo11n.pt        → 检测模型
# - yolo11n-seg.pt    → 分割模型
# - yolo11n-pose.pt   → 姿态模型
# - yolo11n-obb.pt    → 旋转框检测模型

export PYTHONPATH=./
python ./ultralytics/engine/exporter.py
```

执行完成后生成 ONNX 模型，例如：
```
yolo11n.pt → yolo11n.onnx
```

> 使用本仓库的 Ultralytics 确保 ONNX 与 RKNN 官方兼容，避免转换失败。

---

## 3. 配置开发板环境

1. 下载 RKNN 官方仓库：
   - `rknn-toolkit2-2.3.2`（示例版本）
2. 安装相关组件：
   - `rknn-toolkit2` / `rknn-toolkit-lite2` → Python 包
   - `rknpu2` → 驱动（部分板子已预装）
3. 配置 Conda 环境：
   - 安装 `rknn-toolkit2/packages` 和 `rknn-toolkit-lite2/packages` 中对应 Python 的 wheel 文件。
   - 安装对应 `requirements.txt`。

> 完成后，开发板环境即配置完成。

---

## 4. ONNX → RKNN 模型转换

仓库中提供 `RKNN-YOLO11/convert.py`：

```
python convert.py ../model/yolo11n.onnx rk3588
```

- 替换 `../model/yolo11n.onnx` 为自己的 ONNX 模型路径。
- 替换 `rk3588` 为目标平台。

### 可选：量化精度优化

在 `convert.py` 中修改：
```
DATASET_PATH = 'COCO/coco_subset_20.txt'
```

- 替换为自己的校准数据集，提高量化精度。
- 转换完成后生成 RKNN 模型，默认输出：
```
models/yolo11.rknn
```

---

## 5. 测试 RKNN 模型

提供 `test.py` 用于推理测试：

- 原用于安全帽检测模型图片推理。
- 若推理自定义模型，可修改路径和类别配置。
- 推荐使用 **rknn-toolkit-lite2**，依赖少且简单。

---

## 6. 部署流程概览

```
YOLO11 环境配置
        ↓
导出 ONNX 模型
        ↓
开发板 Conda + RKNN 工具包配置
        ↓
ONNX → RKNN 模型量化转换
        ↓
test.py 推理测试
```

完成以上步骤，即可在 RKNN 开发板上部署 YOLO11 模型。

---

## 🔑 小提示

- ONNX 模型必须使用本仓库 Ultralytics 转换，保证与 RKNN 官方兼容。
- 开发板的 Python 环境需与 RKNN 工具包版本对应。
- 量化精度可通过替换校准数据集提高。
- `test.py` 可根据实际模型和数据灵活修改。

## 自述

为方便在RKNN开发板上面部署YOLO11模型，把关键流程自己写了一遍，避免遗忘

首先需要在自己笔记本或者开发板配一遍YOLO11的环境，其实就是把ultralytics和Pytorch配一遍，但是因为RKNN官方为了能够适配他们自身模型的转化，是把ultralytics修改过的，所以你使用官方的ultralytics转化的ONNX模型和采取RKNN的ultralytics转化的ONNX输出是不一样的，大家可以自行使用netron来观察两个onnx的区别
我个人建议按照正常流程配一遍YOLO11的环境，我采取的Anconda环境，把YOLO11的环境配置了一遍，然后pip uninstall ultralytics，把环境里面的ultralytics删除，然后把我仓库的ultralytics_yolo11下载至本地，打开工作空间，在这里面进行转化onnx，具体流程如下

导出onnx模型
在满足 ./requirements.txt 的环境要求后，执行以下语句导出模型

```
调整 ./ultralytics/cfg/default.yaml 中 model 文件路径，默认为 yolo11n.pt，若自己训练模型，请调接至对应的路径。支持检测、分割、姿态、旋转框检测模型。
如填入 yolo11n.pt 导出检测模型
如填入 yolo11n-seg.pt 导出分割模型
如填入 yolo11n-pose.pt 导出姿态模型
如填入 yolo11n-obb.pt 导出OBB模型

export PYTHONPATH=./
python ./ultralytics/engine/exporter.py

执行完毕后，会生成 ONNX 模型. 假如原始模型为 yolo11n.pt，则生成 yolo11n.onnx 模型。

这样调用的U包可以确保为RKNN官方修改后的ultralytics，避免出现ONNX模型输出不一致，导致后期RKNN模型无法使用

当我得到匹配RKNN官方的ONNX模型后，我需要在开发板上面部署环境，首先需要装这些包，自行到rknn的仓库下载rknn-toolkit2-2.3.2，我这边版本是2.3.2

根据需求选择，里面有几个目录rknn-toolkit2、rknn-toolkit-lite2、rknpu2，一般来说，rknpu2是驱动，例如香橙派等开发板都是装好的，但是要是搞的拆机开发板主控芯片是rknn，可能需要安装一下驱动

在开发板上面部署好conda环境后需要在rknn-toolkit2、rknn-toolkit-lite2对应的rknn-toolkit2\packages、rknn-toolkit-lite2\packages里面把对应python的轮子下载下来，并且配置对应requirements.txt环境

这些全部完成，板子上面的环境才配好，接下来要开始ONNX模型量化转化为RKNN模型
我的仓库里面有RKNN-YOLO11/convert.py,
执行命令：python convert.py ../model/yolo11n.onnx rk3588
自己是的onnx模型路径自行替换，转化的平台找到对应的，我这边是3588
为了提升精度，可以换成自己的量化校准测试集，具体替换是convert.py的第四行DATASET_PATH = 'COCO/coco_subset_20.txt'，去对应RRKNN-YOLO11/COCO目录替换为自己的校准数据集
最后会在models文件夹里面生产一个rknn模型，代码默认输出模型名为yolo11.rknn

接下来就是测试模型，由于找了许多推理方法，都感觉不够简洁，最后选择了通过rknn-toolkit-lite2来实现推理，当你环境按照要求配好应该不会缺少rknnlite
我写了一个test.py的推理测试代码，这个是之前用来测试基于YOLO11训练的安全帽模型写的推理图片的代码，如果你们需要推理自己的模型，需要修改一下，整体RKNN部署YOLO11的流程就是这样。
