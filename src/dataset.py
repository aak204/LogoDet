import os
import xml.etree.ElementTree as ET
from collections import OrderedDict
from PIL import Image
import torch
from torch.utils.data import Dataset

class LogoDataset(Dataset):
    """
    Читает подпапки (каждая - это бренд).
    Для каждого .jpg ищет соответствующий .xml, вырезает bndbox (логотип).
    Возвращает (logo_tensor, brand_label).
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.brand_to_idx = OrderedDict()
        idx = 0

        for brand in os.listdir(root_dir):
            brand_path = os.path.join(root_dir, brand)
            if not os.path.isdir(brand_path):
                continue

            if brand not in self.brand_to_idx:
                self.brand_to_idx[brand] = idx
                idx += 1

            for img_file in os.listdir(brand_path):
                if img_file.lower().endswith('.jpg'):
                    img_path = os.path.join(brand_path, img_file)
                    xml_path = os.path.join(
                        brand_path, img_file.replace('.jpg', '.xml'))
                    if os.path.exists(xml_path):
                        self.samples.append(
                            (img_path, xml_path, self.brand_to_idx[brand]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, xml_path, brand_idx = self.samples[idx]
        img = Image.open(img_path).convert('RGB')

        # Парсим XML, получаем bndbox
        tree = ET.parse(xml_path)
        root = tree.getroot()

        obj = root.find('object')
        if obj is None:
            # Если нет объектов, вернём полноразмерную картинку
            logo = img
        else:
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            logo = img.crop((xmin, ymin, xmax, ymax))

        if self.transform:
            logo = self.transform(logo)

        label = torch.tensor(brand_idx, dtype=torch.long)
        return logo, label
