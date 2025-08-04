import os
import random
import shutil

# Caminhos base
base_dir = 'dataset'
images_dir = os.path.join(base_dir, 'images')
labels_dir = os.path.join(base_dir, 'labels')

# Subpastas
train_img = os.path.join(images_dir, 'train')
val_img = os.path.join(images_dir, 'val')
train_lbl = os.path.join(labels_dir, 'train')
val_lbl = os.path.join(labels_dir, 'val')

# Cria pastas se não existirem
for folder in [train_img, val_img, train_lbl, val_lbl]:
    os.makedirs(folder, exist_ok=True)

# Lista todas as imagens da pasta original de treino
image_files = [f for f in os.listdir(train_img) if f.endswith('.jpg')]

# Embaralha e divide
random.seed(42)
random.shuffle(image_files)
split_index = int(len(image_files) * 0.8)
train_files = image_files[:split_index]
val_files = image_files[split_index:]

def mover(imagens, origem_img, origem_lbl, destino_img, destino_lbl):
    for file in imagens:
        name = os.path.splitext(file)[0]
        img_src = os.path.join(origem_img, f"{name}.jpg")
        lbl_src = os.path.join(origem_lbl, f"{name}.txt")

        img_dst = os.path.join(destino_img, f"{name}.jpg")
        lbl_dst = os.path.join(destino_lbl, f"{name}.txt")

        if os.path.exists(img_src):
            shutil.move(img_src, img_dst)
        if os.path.exists(lbl_src):
            shutil.move(lbl_src, lbl_dst)

# Move os arquivos
mover(val_files, train_img, train_lbl, val_img, val_lbl)

print(f'Treinamento: {len(train_files)} imagens')
print(f'Validação:  {len(val_files)} imagens')
