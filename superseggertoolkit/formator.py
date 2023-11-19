from PIL import Image
import os



def keep_every_nth_frame(input_tif_path, output_tif_path, nth_frame=5):
    """
    Keep every nth frame in a multi-frame TIFF file.

    Args:
    input_tif_path (str): Path to the input multi-frame TIFF file.
    output_tif_path (str): Path for the output multi-frame TIFF file with reduced frames.
    nth_frame (int): The interval of frames to keep (default is every 5th frame).
    """
    with Image.open(input_tif_path) as img:
        frames_to_keep = []
        
        for i in range(0, img.n_frames, nth_frame):
            img.seek(i)
            frames_to_keep.append(img.copy())

        frames_to_keep[0].save(output_tif_path, save_all=True, append_images=frames_to_keep[1:])


def image_to_tif_sequence(images_folder):
    tif_images = [f for f in os.listdir(images_folder) if f.lower().endswith('.png')]
    tif_images.sort()

    # Output TIF "video" filename
    output_tif_sequence = images_folder  + 'time_sequences.tif'

    images = []

    for image_filename in tif_images:
        image = Image.open(os.path.join(images_folder,image_filename))
        images.append(image)

    # Save the sequence as a multi-page TIF file
    images[0].save(
        output_tif_sequence,
        save_all=True,
        append_images=images[1:],
        resolution=100.0,  # Set the resolution (DPI)
        compression='tiff_lzw'  # Set the compression method
    )

    print("TIF sequence created successfully.")



def tif_to_gif(tif_filename, gif_filename, duration=150):
    """
    Convert a multi-frame TIF file to a GIF.

    :param tif_filename: Path to the TIF file.
    :param gif_filename: Path where the GIF should be saved.
    :param duration: Duration for each frame in the GIF, in milliseconds.
    """
    with Image.open(tif_filename) as img:
        frames = []

        # Read each frame from the TIF file
        try:
            while True:
                frames.append(img.copy())
                img.seek(img.tell() + 1)
        except EOFError:
            pass

        # Save frames as GIF
        frames[0].save(
            gif_filename,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0
        )

    print(f"GIF saved to {gif_filename}")
