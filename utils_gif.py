from PIL import Image
import os

def combine_gifs_side_by_side(gif1_path, gif2_path, output_path, fps=1.0):
    """
    Combines two GIFs side-by-side into a single GIF.
    """
    print(f"Combining {gif1_path} and {gif2_path} side-by-side...")
    
    gif1 = Image.open(gif1_path)
    gif2 = Image.open(gif2_path)

    frames1 = []
    try:
        while True:
            frames1.append(gif1.copy().convert('RGB'))
            gif1.seek(len(frames1))
    except EOFError:
        pass

    frames2 = []
    try:
        while True:
            frames2.append(gif2.copy().convert('RGB'))
            gif2.seek(len(frames2))
    except EOFError:
        pass

    max_frames = max(len(frames1), len(frames2))
    combined_frames = []

    for i in range(max_frames):
        f1 = frames1[i] if i < len(frames1) else frames1[-1]
        f2 = frames2[i] if i < len(frames2) else frames2[-1]
        
        w = f1.width + f2.width
        h = max(f1.height, f2.height)
        new_frame = Image.new('RGB', (w, h), (255, 255, 255))
        new_frame.paste(f1, (0, 0))
        new_frame.paste(f2, (f1.width, 0))
        combined_frames.append(new_frame)

    duration_ms = int(1000 / fps) if fps > 0 else 1000
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    combined_frames[0].save(output_path, save_all=True, append_images=combined_frames[1:], duration=duration_ms, loop=0)
    print(f"Saved side-by-side visualization to: {output_path}")

