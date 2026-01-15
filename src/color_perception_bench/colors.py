import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from PIL import Image

######################### CONSTANTS #########################
# 949 colors (includes black and white)
XKCD_COLORS_HEX = {k[5:]: v for k, v in mcolors.XKCD_COLORS.items()}

XKCD_COLORS_RGB = {
    name: tuple(int(round(c * 255)) for c in mcolors.to_rgb(hexv))
    for name, hexv in XKCD_COLORS_HEX.items()
}

XKCD_COLORS_HSV = {
    name: tuple(int(round(c * 255)) for c in mcolors.rgb_to_hsv(mcolors.to_rgb(hexv)))
    for name, hexv in XKCD_COLORS_HEX.items()
}


def create_swatch_image(
    color_rgb: tuple[int, int, int], size: int = 244
) -> Image.Image:
    """Create a square swatch image of the given RGB color.

    Args:
        color_rgb: A tuple of (R, G, B) values in the range [0, 255].
        size: The width and height of the square image.

    Returns:
        A NumPy array of shape (size, size, 3) representing the swatch image.
    """
    swatch = np.zeros((size, size, 3), dtype=np.uint8)
    swatch[:, :] = color_rgb
    pil_image = Image.fromarray(swatch)
    return pil_image


def preview_swatch(swatch: Image.Image) -> None:
    """Display the swatch image using matplotlib.

    Args:
        swatch: A PIL Image representing the swatch image.
    """
    fig, ax = plt.subplots(1, 1, figsize=(1, 1))
    ax.imshow(swatch)
    ax.axis("off")
    plt.show()


if __name__ == "__main__":
    import pprint

    # print a small sample from each:
    print("Total colors:", len(XKCD_COLORS_HEX))

    sample_names = list(XKCD_COLORS_HEX.keys())[:5]
    print("XKCD_COLORS_HEX:")
    pprint.pprint({name: XKCD_COLORS_HEX[name] for name in sample_names})

    print("\nXKCD_COLORS_RGB:")
    pprint.pprint({name: XKCD_COLORS_RGB[name] for name in sample_names})

    print("\nXKCD_COLORS_HSV:")
    pprint.pprint({name: XKCD_COLORS_HSV[name] for name in sample_names})

    # create swatch
    red_swatch = create_swatch_image((255, 0, 0))
    print("\nDisplaying red swatch...")
    preview_swatch(red_swatch)
