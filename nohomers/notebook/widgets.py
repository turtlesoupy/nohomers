import base64
from pathlib import Path
from IPython.display import HTML
import ipywidgets as widgets
from PIL import Image
import io
from typing import List, Union


def expand2square(pil_img, background_color=(255, 255, 255), pin="center"):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        if pin == "center":
            result.paste(pil_img, (0, (width - height) // 2))
        elif pin == "corner":
            result.paste(pil_img, (0, 0))
        else:
            raise RuntimeError(f"Bad pin param {pin}")

        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        if pin == "center":
            result.paste(pil_img, ((height - width) // 2, 0))
        elif pin == "corner":
            result.paste(pil_img, (0, 0))
        else:
            raise RuntimeError(f"Bad pin param {pin}")
        return result
    

def resize2square(pil_img, size, background_color=(255, 255, 255), pin="center"):
    new_img = expand2square(pil_img, background_color=background_color, pin=pin)
    return new_img.resize(size, resample=Image.LANCZOS)


def _image_file_formatter(im, thumbnail_size=(256, 256)): 
    if thumbnail_size:
        if im.mode == "RGBA":
            image = Image.new("RGBA", im.size, "WHITE")
            image.paste(im, (0, 0), im)
            im = image.convert("RGB")

        im = expand2square(im, (255, 255, 255))
        im = im.resize(thumbnail_size)

    with io.BytesIO() as buffer:
        im.save(buffer, "png")
        b64 = base64.b64encode(buffer.getvalue()).decode()
    return f'<img src="data:image/jpeg;base64,{b64}">'


def _image_path_formatter(path, thumbnail_size=(256, 256)):
    im = Image.open(path)
    return _image_file_formatter(im, thumbnail_size=thumbnail_size)


def display_pil_images(images, thumbnail_size=(256, 256)):
    return HTML(
        "\n".join(f'<div style="float: left; margin: 10px;"> {_image_file_formatter(im, thumbnail_size)} </div>' for im in images),
    )


def display_df_images(df, cols=["path"], base_path=None, image_size=(128, 128)):
    base_path = Path(base_path) if base_path else None
    return HTML(
        df.iloc[:1000].to_html(
            formatters={
                col: lambda x: _image_path_formatter(
                    base_path / x if base_path else Path(x),
                    image_size,
                )
                for col in cols
            },
            escape=False,
        )
    )


def image_labeller_widget(images: List[Union[str, Path]], write_to_dict, labels, image_size=None):
    def callback(label, key):
        write_to_dict[key] = label
    
    def image_load(path, mode="RGB"):
        with open(path, "rb") as f:
            img = Image.open(f)
            if mode != "RGBA" and img.mode == "RGBA":
                image = Image.new("RGBA", img.size, "WHITE")
                image.paste(img, (0, 0), img) 
                img = image.convert(mode)
            else:
                img = img.convert(mode)

        img = expand2square(img, (255, 255, 255))
        if image_size:
            img = img.resize(image_size)
        return img

    return widgets.VBox(
        [
            labeller_widget(
                image=image_load(image_key),
                options=labels,
                callback=callback,
                callback_data=image_key,
            )
            for image_key in images
        ],
    )


def image_widget_from_pil_image(pil_image):
    img_bytes = None
    with io.BytesIO() as buffer:
        pil_image.save(buffer, "png")
        img_bytes = buffer.getvalue()

    return widgets.Image(
        value=img_bytes,
        format='png',
    )


def labeller_widget(image, options, value=None, callback=None, callback_data=None):
    def handle_change(change):
        val = change['new']
        if callback:
            callback(val, callback_data)

    toggle = widgets.ToggleButtons(
        options=options,
        value=value if value else options[0],
    )

    toggle.observe(handle_change, 'value')

    return widgets.HBox([
        image_widget_from_pil_image(image),
        toggle,
    ])