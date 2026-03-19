import os
import psutil
import pynvml
import torch


# ---------------------------------------------------------------------------------
# from HART: https://github.com/mit-han-lab/hart/blob/main/hart/utils/constants.py
# ---------------------------------------------------------------------------------
# default_prompts = [
#     "beautiful lady, freckles, big smile, blue eyes, short ginger hair, dark makeup, wearing a floral blue vest top, soft light, dark grey background",
#     "anthropomorphic profile of the white snow owl Crystal priestess , art deco painting, pretty and expressive eyes, ornate costume, mythical, ethereal, intricate, elaborate, hyperrealism, hyper detailed, 3D, 8K, Ultra Realistic, high octane, ultra resolution, amazing detail, perfection, In frame, photorealistic, cinematic lighting, visual clarity, shading , Lumen Reflections, Super-Resolution, gigapixel, color grading, retouch, enhanced, PBR, Blender, V-ray, Procreate, zBrush, Unreal Engine 5, cinematic, volumetric, dramatic, neon lighting, wide angle lens ,no digital painting blur."
#     "Bright scene, aerial view, ancient city, fantasy, gorgeous light, mirror reflection, high detail, wide angle lens.",
#     "A 4k dslr image of a lemur wearing a red magician hat and a blue coat performing magic tricks with cards in a garden.",
#     "A silhouette of a grand piano overlooking a dusky cityscape viewed from a top-floor penthouse, rendered in the bold and vivid sytle of a vintage travel poster.",
#     "Crocodile in a sweater.",
#     "Luffy from ONEPIECE, handsome face, fantasy.",
#     "3d digital art of an adorable ghost, glowing within, holding a heart shaped pumpkin, Halloween, super cute, spooky haunted house background.",
#     "an astronaut sitting in a diner, eating fries, cinematic, analog film",
#     "Chinese architecture, ancient style,mountain, bird, lotus, pond, big tree, 4K Unity, octane rendering.",
# ]


default_prompts = [
    'A high-contrast photo of a panda riding a horse. The panda is wearing a wizard hat and is reading a book. The horse is standing on a street against a gray concrete wall. Colorful flowers and the word "PEACE" are painted on the wall. Green grass grows from cracks in the street. DSLR photograph. daytime lighting.',
    'a red cube on the top of blue sphere, the behind is a yellow triangle. A cat is on the left and a dog is on the right',
    'The red hat was on the left of the blue backpack.',
    'woman lying on the beach',
    'woman laying on grass',
    'Epic anime artwork of a wizard atop a mountain at night casting a cosmic spell into the dark sky that says "VAR" made out of colorful energy',
    'A board with text "Hello, VAR"',
    'A paper reads "No!"',
    'A photograph featuring a young woman standing in a field of tall green plants, possibly corn, with the sun shining through the foliage creating a warm, golden glow. The woman is looking off to the side with a gentle expression, and her face is partially obscured by the plants. The sunlight creates a lens flare effect, adding a dreamy quality to the image. The style of the image is naturalistic, capturing a moment of serenity in a rural setting.',
    "A photo-realistic picture. A black and white photograph that captures a man in profile. The man has a beard and mustache, and his hair appears to be swept back. He is wearing a scarf or shawl that is wrapped around his neck and shoulders, adding texture to the image. The photograph is taken from a close-up angle, focusing on the man's face and the upper part of his torso. The lighting is dramatic, with strong contrasts between light and shadow, highlighting the contours of his face and the texture of his hair and clothing. The style of the image is reminiscent of a cinematic or artistic portrait, emphasizing mood and emotion over realism.",
    'A man engaged in the activity of paddleboarding. He is balancing on a white paddleboard with a pink nose, which is partially submerged in the blue water. The man is wearing a black sleeveless top, blue shorts, and sunglasses. His hair is long and appears to be wet, suggesting he has been in the water. He is smiling and seems to be enjoying the moment, with his arms outstretched for balance. The background shows a clear sky and distant mountains, indicating that the setting is likely a large body of water, such as a lake or sea, on a sunny day. The photograph is taken in a realistic style, capturing the action and the natural environment.',
    'a young woman standing in the grass,, in the style of stark black-and-white photography,, hasselblad 1600f,, coastal landscapes,, expressive facial features,, dutch landscapes,, soft atmospheric scenes,, powerful portraits',
    'a digital painting of an old man with a beard and some dark grays,, in the style of photorealistic urban scenes,, uhd image,, algeapunk,, rusty debris,, vibrant portraits,, flickr,, soft-focus portraits',
    'beautiful female warrior,, short blue hair,, shimmering jewels armor,, in the style of Alfons Mucha,, with emphasis on light play and the transparency of the glass,, High and short depth of field,, Ray tracing,, FHD,, hyper quality',
    'a young female hobbit,, ultra realism,, lord of the rings,, snowy forest,, pale hues,, hobbit from lord of the rings who escaped Carn Dum,, grimy,, dirty,, black hair,, homely,, ugly',
    'A dog is walking on a leash with its owner.',
    'A man is running a marathon and crossing the finish line.',
    'an oblong eggplant and a teardrop pear',
    'an oblong cucumber and a teardrop pepper',
    'a brown dog and a blue horse',
    'a rabbit fights with a tiger',
    'three women',
    'three deer',
    'a tree',
    'a photo of a tree',
    'grassland',
    'a woman rides a tiger in the forest',
    'a beautiful scenery area of russia',
    'an oil painting of a house',
    "two girls",
    "three boys",
    'two candles on a marble table next to a silver spoon',
    'woman lying on the beach',
    'woman laying on grass',
    'woman laying on the beach',
    'liberty of statue',
    'a man full body shot',
    'a woman full body shot',
    'a set of sushi which consists of a US map shape',
    'Asian girl near the beach',
    'two women sitting in the sofa and hold red wine cup',
    'a rabbit fights with a tiger',
    'two ninjas fight with each other during night',
    'a red cube on the top of blue sphere, the behind is a yellow triangle. A cat is on the left and a dog is on the right',
    'Epic anime artwork of a wizard atop a mountain at night casting a cosmic spell into the dark sky that says "VAR" made out of colorful energy',
    'a woman having a spa day',
    'two men boxing',
    'a Chinese woman laying on the beach',
    'a man laying on a bed',
    'A brand with "VAR DIT"',
    'A board with text "Hello, VAR"',
    'A paper reads "No!"',
    'A paper reads "VAR Yes!"',
    'American national flag',
    'China national flag',
    'Russia national flag',
    'a woman lying on the beach sunbathing',
    'ironman',
    "Generate the text 'happy' with autumn leaves and cold colors.",
    "Generate the text 'bytedance' with autumn leaves and cold colors.",
    "Generate the text 'GenAI' in a supermarket.",
    "Generate the text 'GenAI' in a grass.",
    "Generate the text 'GenAI' in a ground.",
    "Generate the text 'KCTG' in a book.",
    "Generate the text 'GenAI' in a table.",
    "a Chinese model is sitting on a train, magazine cover, photorealistic, futuristic style",
]


def get_memory_usage(device, nvml_handle):
    """获取当前 CPU 和 GPU 的内存使用情况"""
    process = psutil.Process(os.getpid())
    cpu_mem = process.memory_info().rss / 1024**2 # Bytes to MB

    # PyTorch's perspective
    pytorch_gpu_mem = torch.cuda.memory_allocated(device) / 1024**2 if torch.cuda.is_available() else 0

    # NVIDIA-SMI's perspective
    if nvml_handle:
        info = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle)
        nvsmi_gpu_mem = info.used / 1024**2 # Bytes to MB
    else:
        nvsmi_gpu_mem = 0

    return cpu_mem, pytorch_gpu_mem, nvsmi_gpu_mem


def format_memory(mem_mb):
    """将 MB 格式化为 GB 或 MB"""
    if mem_mb > 1024:
        return f"{mem_mb / 1024:.2f} GB"
    else:
        return f"{mem_mb:.2f} MB"


def get_physical_device_index():
    """
    Gets the physical device index that pynvml can use,
    even when CUDA_VISIBLE_DEVICES is set.
    """
    # Get the logical device index from PyTorch
    logical_device_index = torch.cuda.current_device()

    # Get the CUDA_VISIBLE_DEVICES environment variable
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')

    if cuda_visible_devices:
        # If the variable is set, parse it and find the physical index
        try:
            # Split the comma-separated string into a list of device IDs
            visible_devices = [int(d.strip()) for d in cuda_visible_devices.split(',')]
            # Map the logical index to the physical index
            if logical_device_index < len(visible_devices):
                physical_device_index = visible_devices[logical_device_index]
                return physical_device_index
            else:
                # This case should ideally not happen if PyTorch is configured correctly
                raise IndexError("PyTorch logical device index is out of bounds of CUDA_VISIBLE_DEVICES.")
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not parse CUDA_VISIBLE_DEVICES='{cuda_visible_devices}'. Error: {e}")
            # Fallback to using the logical index directly, which might be incorrect
            return logical_device_index
    else:
        # If the variable is not set, logical and physical indices are the same
        return logical_device_index
