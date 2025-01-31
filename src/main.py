import os
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from dataset import LogoDataset
from model import LogoEncoder
from train import train_and_validate
from utils import show_dataset_samples_with_bbox, verify_logo, collect_reference_embeddings

def main():
    DATA_PATH = "data/LogoDet-3K/Clothes"

    # Загрузка и отображение некоторых образцов
    full_dataset = LogoDataset(DATA_PATH, transform=None)
    show_dataset_samples_with_bbox(full_dataset, 2)

    # Обучение и валидация модели
    train_and_validate()

    # Загрузка модели
    model = LogoEncoder(emb_dim=1024)
    ckpt = torch.load("lcnet_logo_encoder.pth", map_location='cuda')
    model.load_state_dict(ckpt)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.cuda().eval()

    brandX_imgs = [
        "data/LogoDet-3K/Clothes/Berghaus/0.jpg",
        "data/LogoDet-3K/Clothes/Berghaus/1.jpg"
    ]

    # Собираем эмбеддинги эталонов
    reference_embs = collect_reference_embeddings(brandX_imgs, model)

    print(f"Collected {len(reference_embs)} reference emb(s) for brand 'Berghaus'")

    queries = [
        "data/LogoDet-3K/Clothes/Berghaus/10.jpg",
        "data/LogoDet-3K/Clothes/Boboli/1.jpg"
    ]

    for query_path in queries:
        if not os.path.exists(query_path):
            print(f"[WARNING] Query file not found: {query_path}")
        else:
            query_img = Image.open(query_path).convert("RGB")
            is_brandX, sim_val = verify_logo(query_img, reference_embs, model, threshold=0.65)
            print(f"\nQuery: {query_path}")
            print(f"Is brand Berghaus? {is_brandX}. Mean similarity = {sim_val:.2f}")

if __name__ == "__main__":
    main()
