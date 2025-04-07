import configparser
import os
import json
import re
import subprocess
import numpy as np
import cv2
from PIL import Image


def sanitize_filename(filename):
    """清理文件名并强制使用.png扩展名"""
    name = os.path.splitext(filename)[0]  # 分离原始扩展名
    clean_name = re.sub(r"[^\w-]", "", name.replace(' ', '_'))
    return f"{clean_name}.png".lower()  # 强制使用.png扩展名


def crop_to_content_cv2(img_np):
    """使用OpenCV裁剪透明边框"""
    if img_np.shape[2] == 4:
        alpha = img_np[:, :, 3]
    else:
        alpha = np.full(img_np.shape[:2], 255, dtype=np.uint8)
    
    coords = cv2.findNonZero(alpha)
    if coords is None:
        return img_np
    
    x, y, w, h = cv2.boundingRect(coords)
    return img_np[y:y+h, x:x+w]


def process_image_cv2(img_np, autofit):
    """OpenCV处理核心逻辑（保持原有功能不变）"""
    target_size = (650, 900)
    img_np = crop_to_content_cv2(img_np)
    
    h, w = img_np.shape[:2]
    target_w, target_h = target_size
    
    if autofit:
        return cv2.resize(img_np, target_size, interpolation=cv2.INTER_AREA)
    else:
        scale = min(target_w / w, target_h / h)
        new_size = (int(w * scale), int(h * scale))
        resized = cv2.resize(img_np, new_size, interpolation=cv2.INTER_AREA)
        
        canvas = np.zeros((target_h, target_w, 4), dtype=np.uint8)
        y_offset = (target_h - new_size[1]) // 2
        x_offset = (target_w - new_size[0]) // 2
        
        if resized.shape[2] == 4:
            alpha = resized[:, :, 3] / 255.0
            for c in range(3):
                canvas[y_offset:y_offset+new_size[1], x_offset:x_offset+new_size[0], c] = \
                    resized[:, :, c] * alpha + canvas[y_offset:y_offset+new_size[1], x_offset:x_offset+new_size[0], c] * (1 - alpha)
            canvas[y_offset:y_offset+new_size[1], x_offset:x_offset+new_size[0], 3] = \
                np.maximum(resized[:, :, 3], canvas[y_offset:y_offset+new_size[1], x_offset:x_offset+new_size[0], 3])
        else:
            canvas[y_offset:y_offset+new_size[1], x_offset:x_offset+new_size[0], :3] = resized
            canvas[y_offset:y_offset+new_size[1], x_offset:x_offset+new_size[0], 3] = 255
        
        return canvas


def load_image(path):
    """加载图像并确保RGBA格式"""
    img_np = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img_np is None:
        raise ValueError(f"无法读取图像: {path}")
    
    # 转换通道顺序并添加Alpha通道
    if img_np.shape[2] == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGBA)
    elif img_np.shape[2] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGRA2RGBA)
    return img_np


def save_image_png(img_np, path, quality, use_pngquant=False, colors=256):
    """确保保存为PNG格式"""
    try:
        # 强制.png扩展名
        path = os.path.splitext(path)[0] + ".png"
        
        # 转换到Pillow格式
        if img_np.shape[2] == 4:
            img_pil = Image.fromarray(img_np, 'RGBA')
        else:
            img_pil = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_RGB2RGBA))
        
        temp_path = path + ".tmp.png"
        
        # 压缩逻辑保持不变
        if quality < 100:
            alpha = img_pil.split()[-1]
            rgb = img_pil.convert('RGB')
            pillow_colors = max(2, int(256 * quality / 100))
            quant = rgb.quantize(colors=pillow_colors, method=Image.MEDIANCUT)
            quant_rgba = quant.convert('RGBA')
            quant_rgba.putalpha(alpha)
            quant_rgba.save(temp_path, 'PNG', optimize=True, compress_level=9)
        else:
            img_pil.save(temp_path, 'PNG', optimize=True)

        # pngquant处理
        if use_pngquant:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            pngquant_path = os.path.join(script_dir, 'pngquant.exe')
            
            cmd = [
                pngquant_path,
                str(colors),
                '--quality', '60-90',
                '--speed', '3',
                '--strip',
                '--force',
                '--output', path,
                temp_path
            ]
            
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            
            if result.returncode not in [0, 98]:
                print(f"pngquant错误: {result.stderr.decode('utf-8')}")
                os.replace(temp_path, path)
            else:
                os.remove(temp_path)
        else:
            os.replace(temp_path, path)
            
    except Exception as e:
        print(f"图片保存失败: {str(e)}")
        if os.path.exists(temp_path):
            os.remove(temp_path)



def main():
    # ========== 依赖检查 ==========
    try:
        from PIL import Image
    except ImportError:
        print("错误：需要安装Pillow库。请运行 pip install Pillow")
        return

    # ========== 配置处理 ==========
    config = configparser.ConfigParser()
    default_config = {
        'ResourcePack': {
            'start_id': '1',
            'namespace': 'card',
            'output_dir': 'CARD_RP',
            'pack_format': '46',
            'description': 'A Minecraft card game resource pack.',
            'autofit': 'true',
            'mode': 'id',
            'compress': '100',
            'pngquant': 'false',
            'pngquant_color': '256'
        }
    }
    
    if not os.path.exists('config.ini'):
        config.read_dict(default_config)
        with open('config.ini', 'w') as f:
            config.write(f)
        print("已创建默认config.ini文件")

    config.read('config.ini')
    conf = config['ResourcePack']

    # 读取基础配置
    start_id = int(conf.get('start_id', '1000'))
    namespace = conf.get('namespace', 'card')
    output_dir = conf.get('output_dir', 'CARD_RP')
    pack_format = int(conf.get('pack_format', '46'))
    description = conf.get('description', 'A Minecraft card game resource pack.')
    autofit = conf.get('autofit', 'true').lower() == 'true'

    # 读取模式配置
    mode = conf.get('mode', 'id').lower()
    if mode not in ['id', 'name']:
        mode = 'id'
        print("警告：无效的mode配置，使用默认值 id")

    # 读取压缩配置
    compress_quality = int(conf.get('compress', '100'))
    compress_quality = max(0, min(compress_quality, 100))
    
    # 读取pngquant配置
    use_pngquant = conf.get('pngquant', 'false').lower() == 'true'
    pngquant_colors = int(conf.get('pngquant_color', '256'))

    # ========== 初始化变量 ==========
    name_mapping = {}
    current_id = start_id

    # ========== 目录结构创建 ==========
    os.makedirs('images', exist_ok=True)
    assets_path = os.path.join(output_dir, 'assets', namespace)
    for dir_path in [
        os.path.join(assets_path, 'items'),
        os.path.join(assets_path, 'models', 'item'),
        os.path.join(assets_path, 'textures', 'item')
    ]:
        os.makedirs(dir_path, exist_ok=True)

  # ========== 图片处理 ==========
    has_back = False
    valid_ext = ('.png', '.jpg', '.jpeg')
    image_files = [f for f in os.listdir('images') if f.lower().endswith(valid_ext)]

    # 处理 back.png（使用OpenCV）
    back_files = [f for f in image_files if os.path.splitext(f)[0].lower() == 'back']
    if back_files:
        back_source = back_files[0]
        try:
            img_np = load_image(os.path.join('images', back_source))
            processed = process_image_cv2(img_np, autofit)
            back_name = 'back.png'
            back_path = os.path.join(assets_path, 'textures', 'item', back_name)
            save_image_png(
                processed,
                back_path,
                compress_quality,
                use_pngquant=use_pngquant,
                colors=pngquant_colors
            )
            print(f"已处理背景图片：{back_source} → {back_name}")
            image_files.remove(back_source)
            has_back = True
        except Exception as e:
            print(f"背景图片处理错误：{str(e)}")
    else:
        print("警告：未找到 back.png！将使用卡片正面作为背面纹理")

    # 处理其他图片（使用OpenCV加速）
    texture_count = 0
    total = len(image_files)
    
    for idx, filename in enumerate(image_files, 1):
        try:
            img_np = load_image(os.path.join('images', filename))
            processed = process_image_cv2(img_np, autofit)
            
            if mode == 'id':
                output_name = f"{current_id}.png"
                current_id += 1
            else:
                output_name = sanitize_filename(filename)
                name_mapping[output_name] = {
                    'original': filename,
                    'clean_name': os.path.splitext(output_name)[0]
                }
            
            output_path = os.path.join(assets_path, 'textures', 'item', output_name)
            save_image_png(
                processed,
                output_path,
                compress_quality,
                use_pngquant,
                pngquant_colors
            )
            
            print(f"进度：{idx}/{total} → {output_name}")
            texture_count += 1
        except Exception as e:
            print(f"图片处理失败：{filename} - {str(e)}")

    # ========== 生成JSON文件 ==========
    MODEL_TEMPLATE = {
        "credit": "Made with Blockbench",
        "texture_size": [650, 900],
        "textures": {"0": f"{namespace}:item/back", "1": "{texture_id}", "particle": f"{namespace}:item/back"},
        "elements": [{"from": [2,0,8],"to": [16,18,8],"rotation": {"angle":0,"axis":"x","origin":[9,9,8]},
            "faces": {"north":{"uv":[16,16,0,0],"rotation":180,"texture":"#0"},
                      "south":{"uv":[0,0,16,16],"texture":"#1"}}}],
        "gui_light": "front",
        "display": {"thirdperson_righthand": {"rotation":[49.5,0,0],"translation":[0,1,1.25],"scale":[0.20703,0.20703,0.20703]},
                    "thirdperson_lefthand": {"rotation":[49.5,0,0],"translation":[0,1,1.25],"scale":[0.20703,0.20703,0.20703]},
                    "firstperson_righthand":{"translation":[-10,8,-4]},"firstperson_lefthand":{"translation":[-10,8,-4]},
                    "ground":{"translation":[0,1.75,0],"scale":[0.2,0.2,1]},"gui":{"translation":[-1,0,0]},
                    "fixed":{"rotation":[0,180,0],"translation":[1.25,-1,0.5]}}}

    # 生成item模型JSON
    for item_info in (range(start_id, start_id + texture_count) if mode == 'id' else name_mapping.values()):
        if mode == 'id':
            item_id = item_info
            model_name = str(item_id)
            item_path = os.path.join(assets_path, 'items', f'{item_id}.json')
        else:
            model_name = item_info['clean_name']
            item_path = os.path.join(assets_path, 'items', f'{model_name}.json')

        with open(item_path, 'w') as f:
            json.dump({
                "model": {"type": "minecraft:model", "model": f"{namespace}:item/{model_name}"}
            }, f, indent=4)

        # 生成3D模型JSON
        model_data = MODEL_TEMPLATE.copy()
        base_texture = f"{namespace}:item/{model_name}"
        if not has_back:
            model_data["textures"]["0"] = base_texture
            model_data["textures"]["particle"] = base_texture
        model_data["textures"]["1"] = base_texture
        
        model_path = os.path.join(assets_path, 'models', 'item', f'{model_name}.json')
        with open(model_path, 'w') as f:
            json.dump(model_data, f, indent=4)

    # 创建pack.mcmeta
    pack_meta = {"pack": {"pack_format": pack_format, "description": description}}
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'pack.mcmeta'), 'w') as f:
        json.dump(pack_meta, f, indent=4)

    print(f"资源包生成完成！保存路径：{os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()