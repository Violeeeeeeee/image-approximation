import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import os


def img_is_color(img):

    if len(img.shape) == 3:
        # Check the color channels to see if they're all the same.
        c1, c2, c3 = img[:, : , 0], img[:, :, 1], img[:, :, 2]
        if (c1 == c2).all() and (c2 == c3).all():
            return True

    return False

def show_image_list(list_images, list_titles=None, list_cmaps=None, grid=True, num_cols=2, figsize=(20, 10), title_fontsize=30):
    '''
    Shows a grid of images, where each image is a Numpy array. The images can be either
    RGB or grayscale.

    Parameters:
    ----------
    images: list
        List of the images to be displayed.
    list_titles: list or None
        Optional list of titles to be shown for each image.
    list_cmaps: list or None
        Optional list of cmap values for each image. If None, then cmap will be
        automatically inferred.
    grid: boolean
        If True, show a grid over each image
    num_cols: int
        Number of columns to show.
    figsize: tuple of width, height
        Value to be passed to pyplot.figure()
    title_fontsize: int
        Value to be passed to set_title().
    '''

    assert isinstance(list_images, list)
    assert len(list_images) > 0
    assert isinstance(list_images[0], np.ndarray)

    if list_titles is not None:
        assert isinstance(list_titles, list)
        assert len(list_images) == len(list_titles), '%d imgs != %d titles' % (len(list_images), len(list_titles))

    if list_cmaps is not None:
        assert isinstance(list_cmaps, list)
        assert len(list_images) == len(list_cmaps), '%d imgs != %d cmaps' % (len(list_images), len(list_cmaps))

    num_images  = len(list_images)
    num_cols    = min(num_images, num_cols)
    num_rows    = int(num_images / num_cols) + (1 if num_images % num_cols != 0 else 0)

    # Create a grid of subplots.
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    # Create list of axes for easy iteration.
    if isinstance(axes, np.ndarray):
        list_axes = list(axes.flat)
    else:
        list_axes = [axes]

    for i in range(num_images):

        img    = list_images[i]
        title  = list_titles[i] if list_titles is not None else 'Image %d' % (i)
        cmap   = list_cmaps[i] if list_cmaps is not None else (None if img_is_color(img) else 'gray')

        list_axes[i].imshow(img, cmap=cmap)
        list_axes[i].set_title(title, fontsize=title_fontsize)
        list_axes[i].grid(grid)

    for i in range(num_images, len(list_axes)):
        list_axes[i].set_visible(False)

    fig.tight_layout()
    _ = plt.show()
    fig.savefig('plot.png')


def load_images_from_folder(folder_path):
    images = []
    titles = []

    if not os.path.exists(folder_path):
        print(f"Папка {folder_path} не знайдена!")
        return [], []

    file_list = sorted(os.listdir(folder_path))

    for filename in file_list:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            path = os.path.join(folder_path, filename)

            img = cv2.imread(path)

            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                images.append(img)
                titles.append(filename)

    print(f"{len(images)} images from {folder_path}")
    return images, titles



# how to use:
# imgs, titles = load_with_glob("mona_0/gen_*.png")
def main():
    # img1 = cv2.imread("data/power_128.jpg")

    folder = "mona3/c"

    imgs, names = load_images_from_folder(folder)

    if len(imgs) > 0:
        # step = max(1, len(imgs) // 6)
        # subset_imgs = imgs[::step]

        show_image_list(imgs,
        # show_image_list(subset_imgs,
                        # list_titles=['0 step', '0 step', '0 step', '0 step', '0 step', '1000 step', '1000 step', '1000 step',
                        #              '1000 step', '1000 step', '5000 step', '5000 step', '5000 step', '5000 step', '5000 step',
                        #              '10000 step', '10000 step', '10000 step', '10000 step', '10000 step',
                        #              '999999 step', '99999 step', '19999 step', '19999 step', '19999 step'],
                        # names,
                        # list_titles=['3000 step', '10000 step', '20000 step', '39999 step'],
                        list_titles=['mona0', 'mona1', 'mona2', 'mona3', 'mona4'],
                        # list_titles=['original 128x128', 'mona3 39999 step'],
                        num_cols=2,
                        figsize=(12, 4),
                        grid=False,
                        title_fontsize=10)





if __name__ == "__main__":
    main()
