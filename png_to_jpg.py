from PIL import Image
import glob

images_paths = list(glob.glob('images/*.png'))
#print("images/jpg_images/" + (images_paths[0][:-4] + ".jpg")[7:])


for img_path in images_paths:
    im1 = Image.open(img_path)
    im1.save("images/jpg_images/" + (img_path[:-4] + ".jpg")[7:])