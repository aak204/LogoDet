import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torchvision import transforms
from collections import OrderedDict
import xml.etree.ElementTree as ET

def verify_logo(query_img, reference_embs, model, threshold=0.5):
    """
    :param query_img: PIL.Image (кроп логотипа)
    :param reference_embs: list[torch.Tensor], где каждый [EMB_DIM], нормирован
    :param model: обученный LogoEncoder (уже .eval() и .cuda())
    :param threshold: порог косинусного сходства
    :return: (bool, float) - флаг "это бренд" и усреднённое сходство
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    model.eval()
    with torch.no_grad():
        q_tensor = transform(query_img).unsqueeze(0).cuda()
        q_emb = model(q_tensor).squeeze(0)  # [EMB_DIM]

        sims = []
        for ref_emb in reference_embs:
            # Убедимся, что ref_emb на GPU:
            if ref_emb.device != q_emb.device:
                ref_emb = ref_emb.to(q_emb.device)
            cos_sim = torch.dot(q_emb, ref_emb).item()  # уже нормированы => косинус
            sims.append(cos_sim)

    mean_sim = float(np.mean(sims))
    return (mean_sim >= threshold), mean_sim 

def invert_dict(d):
    return {v: k for k, v in d.items()}

def show_dataset_samples_with_bbox(dataset, n=5):
    """
    Выбирает случайные n примеров из dataset.samples,
    отображает исходное изображение с нарисованным bbox,
    а также выводит название бренда в заголовке.
    """
    # Если датасет пуст
    if len(dataset.samples) == 0:
        print("No samples in the dataset!")
        return
    
    # Создаём обратный словарь idx -> brand_name,
    # чтобы получить текстовое название бренда по индексу
    idx_to_brand = invert_dict(dataset.brand_to_idx)
    
    # Случайные индексы из выборки
    indices = random.sample(range(len(dataset.samples)), min(n, len(dataset.samples)))
    
    fig, axes = plt.subplots(1, len(indices), figsize=(6 * len(indices), 6))
    if len(indices) == 1:
        # Если n=1, чтобы axes не был массивом
        axes = [axes]
    
    for ax, i in zip(axes, indices):
        img_path, xml_path, brand_idx = dataset.samples[i]
        
        # Парсим XML, чтобы получить bounding box
        tree = ET.parse(xml_path)
        root = tree.getroot()
        obj = root.find('object')
        
        img = Image.open(img_path).convert('RGB')
        ax.imshow(img)
        
        brand_name = idx_to_brand[brand_idx]
        
        if obj is not None:
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            
            # Рисуем прямоугольник
            rect = patches.Rectangle(
                (xmin, ymin),  # (x, y) – левый верхний угол
                xmax - xmin,   # ширина
                ymax - ymin,   # высота
                linewidth=2,
                edgecolor='red',
                facecolor='none'
            )
            ax.add_patch(rect)
            
            ax.set_title(f"Brand: {brand_name}\n({img_path})", fontsize=10)
        else:
            ax.set_title(f"Brand: {brand_name}\n(No object found)", fontsize=10)
        
        ax.axis('off')
    
    plt.tight_layout()
    plt.show() 

def collect_reference_embeddings(image_paths, model):
    """
    Собирает эмбеддинги для списка эталонных изображений.
    """
    infer_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    reference_embs = []
    with torch.no_grad():
        for path in image_paths:
            if not os.path.exists(path):
                print(f"[WARNING] File not found: {path}")
                continue
            rimg = Image.open(path).convert("RGB")
            rtensor = infer_transform(rimg).unsqueeze(0).cuda()
            emb = model(rtensor).squeeze(0)  # [emb_dim]
            reference_embs.append(emb)

    brand_name = os.path.basename(os.path.dirname(image_paths[0]))
    print(f"Collected {len(reference_embs)} reference emb(s) for brand '{brand_name}'")
    return reference_embs
