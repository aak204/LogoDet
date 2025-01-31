import torch

def validate_simple(model, val_loader):
    """
    - Прогоняем батчи
    - Внутри батча ищем "ближайшего соседа" по косинусной похожести (кроме самого себя)
    - Считаем, сколько совпало меток.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.cuda()
            labels = labels.cuda()
            feats = model(imgs)  # [B, EMB_DIM], уже нормированы

            sim_matrix = torch.matmul(feats, feats.T)  # [B, B]
            batch_size = feats.size(0)

            for i in range(batch_size):
                # Зануляем самосходство:
                sim_matrix[i, i] = -999.0
                nn_idx = torch.argmax(sim_matrix[i])
                if labels[nn_idx] == labels[i]:
                    correct += 1
            total += batch_size

    return correct / total
