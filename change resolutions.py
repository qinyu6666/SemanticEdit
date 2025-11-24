import cv2

# 读取图片
image_path = './data/newsize/33.jpg'  # 替换为你的图片路径
image = cv2.imread(image_path)

# 检查图片是否正确读取
if image is None:
    print("图片读取失败，请检查路径是否正确")
else:
    # 修改分辨率
    new_image = cv2.resize(image, (232,295), interpolation=cv2.INTER_AREA)

    # 保存图片
    save_path = './results/newsized/33.jpg'  # 替换为你想要保存新图片的路径
    cv2.imwrite(save_path, new_image)

    print(f"图片已保存到：{save_path}")