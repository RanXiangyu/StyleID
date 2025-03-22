from PIL import Image
import torchvision.transforms as transforms

def crop_image_to_512(image_path, output_path):
    """
    将指定图片裁剪成512x512的大小并保存到输出路径
    :param image_path: 输入图片路径
    :param output_path: 输出图片路径
    """
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.CenterCrop(512),
        transforms.Resize((512, 512), transforms.InterpolationMode.LANCZOS)
    ])
    cropped_image = transform(image)
    cropped_image.save(output_path)

# 示例用法
if __name__ == "__main__":
    input_image_path = "/data2/ranxiangyu/styleid_out/style/masson.jpeg"
    output_image_path = "/data2/ranxiangyu/styleid_out/style/masson1.jpeg"
    crop_image_to_512(input_image_path, output_image_path)