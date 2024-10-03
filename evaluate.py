import clip
import torch
from PIL import Image

render_path = "E:\code\gaussian-splatting\output\\room\clg_low\\test\ours_7000\\renders\\"
gt_path = "E:\code\gaussian-splatting\output\\room\clg_low\\test\ours_7000\gt\\"

# print(render_path,gt_path)

device = "cuda"
model, preprocess = clip.load("ViT-B/32", device="cuda")

filename1 = "00034.png"
filename2 = "2.png"

image1 = preprocess(Image.open(render_path + filename1)).unsqueeze(0).to(device)
image2 = preprocess(Image.open(render_path + filename2)).unsqueeze(0).to(device)

with torch.no_grad():
    image1_feature = model.encode_image(image1)
    image2_feature = model.encode_image(image1)
    q = torch.nn.functional.cosine_similarity(image1_feature, image2_feature, dim=1)

print(q)
print(image1_feature.size())