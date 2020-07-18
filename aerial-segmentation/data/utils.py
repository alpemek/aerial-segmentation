from PIL import Image
import numpy as np


def overlay(image, mask, alpha=1):
    """
    Args:
        image (string): path to image file
        mask (string): path to mask file
        alpha (float): value between 0 and 1 specifying alpha
    """
    assert 0 <= alpha <= 1
    img = Image.open(image)
    mask_img = Image.open(mask)
    img = img.convert("RGBA")
    mask_img = mask_img.convert("RGBA")

    new_data = []
    for item in mask_img.getdata():
        if item == (255, 255, 255, 255):
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append((*item[:-1], int(alpha*255)))

    mask_img = Image.new('RGBA', mask_img.size)
    mask_img.putdata(new_data)
    img.alpha_composite(mask_img, (0, 0)) # Same result as in the example
    img.show()


if __name__ == "__main__":
    dataset_name = "paris"
    image_id = 6
    image = "../../dataset/{0}/{0}{1}_image.png".format(dataset_name, image_id)
    mask = "../../dataset/{0}/{0}{1}_labels.png".format(dataset_name, image_id)
    alpha = 0.5
    overlay(image, mask, alpha)