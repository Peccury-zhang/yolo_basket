from ultralytics import YOLO

# Load a model
# model = YOLO("yolo26n.yaml")  # build a new model from YAML
model = YOLO("yolo11s-seg.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolo26n.yaml").load("yolo26n.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="yolo11s-seg.yaml", 
                      epochs=100, #epochs训练轮数
                      imgsz=640, #imgsz训练图像大小
                      workers=0, # 加速训练 默认8
                      device="0",# gpu设备,用多少块显卡，0是默认用1个
                      batch=0.8, #批处理大小。默认16，表示一次训练8张图。怕显存不够，可以设置为0.8，表示显存最大用到80%
                      ) 

# if __name__ == "__main__":
#     model = YOLO('weights/raw/yolo11s-seg.pt')  # load a pretrained model (recommended for training)
#     # model = YOLO('D:/PycharmProjects/4.YOLO/yolo_final/runs/segment/train2/weights/best.pt')  # load a pretrained model (recommended for training)
#     model.train(data='A_seg.yaml',
#                 # cache=False,#缓存图片 默认True
#                 epochs=150,#训练轮数
#                 imgsz=640,#图片大小 默认640
#                 batch=8,#批处理大小 默认8
#                 # patience=30,#早停 默认为100
#                 # close_mosaic=10,#设置在多少轮前关闭mosaic数据增强 默认为None
#                 workers=0,#加速训练 默认8
#                 device="0",# gpu设备
#                 amp=False  # 关闭AMP
#                 )