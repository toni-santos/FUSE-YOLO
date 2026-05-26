# FUSE-YOLO 🌀

FUSE-YOLO is a generic YOLOv5 adaptation for multispectral object detection tasks. This model uses Ultralytics [YOLOv5](https://github.com/ultralytics/yolov5/) as a basis, adapting and building upon this model to allow for any early or late feature fusion task.

---

### Fusion Methodologies 🔨

We have implemented 2 fusion methodologies:

- **Early fusion**
- **Late fusion**

The terminology of early and late is related to the backbone, meaning that early fusion is done prior to the backbone and late fusion after it.

### Fusion Modules 🧩

3 fusion modules have been developed and implemented, in increasingly complexity order:

- **CatFuse** - Simple concatenation
- **CBMAC** - CBAM-based fusion
- **TransEnc** - Utilizing a transformer's encoder

### Developing New Modules ⭐

If you wish to add or develop a new module please follow these steps:

1. Add your module to `models/fuse.py`
2. Properly add it to the `models/yolo.py` parsing function 
3. Call it in your configuration file

Calling the fusion module in your configuration file is done similarly to any other module, please be sure to consult an example, such as `models/early_catfuse.yaml`.

### Citing

A paper based on the work of the dissertation that is behind this model has been published. The paper is available [here](https://link.springer.com/article/10.1007/s00138-026-01807-y).
