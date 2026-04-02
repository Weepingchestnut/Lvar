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

default_video_prompts = [
    # ------ from VBench ------
    # temporal_flickering
    'In a still frame, a stop sign stands prominently against a clear blue sky backdrop. The octagonal red sign with bold white letters is positioned slightly off-center, drawing attention with its vivid color. Sunlight casts a soft shadow across the sign’s surface, emphasizing its edges and creating a subtly dramatic effect. The scene is calm, with no visible movement, and the simplicity of the sign against the expansive sky highlights its importance and message. The camera captures the scene from a slightly low angle, adding depth and perspective to the composition.',
    'A close-up view of a toilet frozen in time, capturing water defying gravity as it simulates mid-flush, forming splashes in mid-air. The toilet, with a gleaming white porcelain surface, is illuminated by bright, overhead lighting, creating a stark contrast with the water droplets. The camera slowly circles the scene, offering a dynamic perspective of the suspended motion. The play of light on the droplets creates a sparkling effect, highlighting the stillness of the moment despite the dynamic action typically associated with flushing.',
    "A sleek, modern laptop is seen frozen in time on a minimalist desk. The screen displays a paused vibrant image of a cityscape at night, with colorful lights illuminating the scene. The laptop's lid is open at a typical viewing angle, and the keyboard glows softly with backlighting. The trackpad is centered below the keyboard, and the device's surface reflects a bit of ambient light. A subtle play of shadows and lights enhances the scene, capturing the stillness of the moment. The camera gently circles around the laptop, showcasing its elegance from different angles.",
    'A narrow, cobblestone alleyway is depicted, nestled between tall, old stone buildings. The alley is bathed in gentle, warm sunlight filtering through, casting soft shadows that add depth to the scene. The buildings are adorned with small, ornate wrought-iron balconies and flower boxes that overflow with vibrant blooms. At the end of the alley, a lush vine climbs the wall, adding greenery to the serene setting. The camera pans slowly from the ground up, capturing the textured stones and the charming architectural details, enhancing the peaceful and picturesque atmosphere.',
    'The scene captures a tranquil tableau of a stylish bar, featuring dim, ambient lighting and a warm, inviting atmosphere. Behind the bar, an elegant display of bottles glistens reflectively under the soft overhead lights. A row of bar stools with plush cushioning stands in front of the counter, adding to the cozy environment. The bartender, a handsome man with neatly styled hair and a crisp white shirt, is seen polishing a glass while offering a welcoming smile. The camera gently pans across the bar, highlighting the sophisticated setting and capturing the serene vibe of this refined space.',
    # multiple_objects
    'A bird is perched on the branch of a tree, with its vibrant feathers illuminated by the warm sunlight filtering through the leaves. Below, a cat prowls stealthily across the grass, its eyes focused on the bird above. The cat, with sleek fur and a curious expression, moves cautiously forward, tail swaying slightly. Meanwhile, the bird occasionally glances down, its feathers ruffling slightly in the gentle breeze. The scene is set in a lush garden, with colorful flowers and dappled light creating a serene atmosphere. The camera slowly pans upward from the cat to the bird, capturing the graceful movements of both animals.',
    "A charming cat and a playful dog are in a cozy living room, lounging on a soft carpet. The cat, with its sleek fur and bright eyes, is gently batting its paw at a small toy mouse. Meanwhile, the dog, with its fluffy coat and attentive expression, watches the cat's playful antics with curiosity. The setting is warm and inviting, with soft sunlight filtering in through a nearby window, casting a gentle glow on the carpet where the animals are. The gentle sway of the curtains adds to the calming atmosphere. The camera captures this heartwarming scene with a slow, smooth pan across the room, highlighting the interaction between the two animals.",
    "On a lush green field, a beautiful dog is playfully running around a majestic horse. The dog has a sleek coat and a joyful expression, its tail wagging energetically. The horse, with a glossy mane and a strong, elegant stature, stands calmly, occasionally glancing at the lively dog. The scene has a serene and harmonious atmosphere. The camera captures a wide shot, moving slightly to follow the dog's movement, emphasizing the peaceful interaction between the two animals.",
    'A majestic horse stands beside a fluffy sheep in a lush, green pasture under a bright, clear sky. The horse, with its shiny brown coat and flowing mane, stands tall and gracefully, staring into the distance. The sheep, with its woolly white fleece, grazes contentedly on the grass beside the horse. The scene is peaceful and pastoral, with a gentle breeze rustling the grass. The camera slowly pans from the horse to the sheep, highlighting their companionship and the serene countryside setting.',
    'In a sunny, open pasture filled with lush green grass, a sheep and a cow stand grazing side by side. The sheep, with its fluffy white wool, is nibbling on the grass, while the cow, with its sleek brown and white coat, chews contentedly nearby. The bright daylight creates a serene and peaceful atmosphere, with a few small daisies scattered around their feet. The camera captures this idyllic scene from a slightly elevated angle, gradually zooming in to focus on the gentle interaction between the two animals.',
    # human_action
    'A young man is riding a bike along a scenic pathway, with lush green trees lining both sides. He is wearing a casual t-shirt and shorts, and his hair flows gently in the breeze as he pedals forward. The sun filters through the trees, creating a dappled light effect on the path. The camera follows his movement smoothly, capturing the rhythm of his pedaling and the serene, natural surroundings.',
    'A person is marching with purposeful strides along a paved walkway in a park, surrounded by lush greenery. He is a handsome young man in his twenties, wearing a crisp white shirt and dark trousers. His expression is focused and determined, and his posture is upright, conveying confidence. The camera follows him, slightly elevated, capturing his rhythmic movements as he marches forward towards the camera, while the tree-lined path creates a serene and natural backdrop.',
    'A young, athletic woman is roller skating gracefully on a smooth path. She has long flowing hair and wears a stylish, fitted tank top and shorts. As she skates towards the camera, her movements are fluid and controlled, effortlessly gliding forward with a joyful expression on her face. The sun casts a warm, golden glow, creating a lively and energetic atmosphere. The camera smoothly follows her motion, capturing her skilled and elegant skating.',
    'A man is tasting beer in a cozy pub setting with warm ambient lighting. He is handsome, with short, neatly styled hair, and is dressed casually in a dark shirt. The man lifts a clear glass of amber beer to his lips, savoring the aroma before taking a small sip. His expression is thoughtful and appreciative as he tastes the beer, and he slightly nods as if savoring the flavor. The scene is framed with wooden tables and shelves filled with various bottles in the background, adding to the inviting and relaxed atmosphere. The camera slowly zooms in, capturing the thoughtful expression on his face.',
    'A young woman is standing and clapping her hands with a bright smile. She has long, wavy hair cascading down her shoulders and is wearing an elegant blue dress that accentuates her charm. Her eyes are sparkling with joy as she applauds enthusiastically. The camera captures her from the front, zooming in slightly to highlight her expressive facial features and graceful movements, creating a warm and celebratory atmosphere.',
    # subject_consistency, dynamic_degree, motion_smoothness
    'A woman is gracefully swimming in the ocean, her long hair flowing behind her as she moves through the water. The sunlight sparkles on the ocean surface, creating a shimmering effect. She wears a bright-colored swimsuit that stands out against the blue of the water. Her movements are smooth and fluid, with her arms cutting through the waves as she swims towards the horizon. The camera follows her from a low angle, capturing the gentle rise and fall of the ocean around her, enhancing the serene and tranquil atmosphere of the scene.',
    'A confident man, appearing to be in his early thirties, stands at the front of a modern conference room, delivering a presentation to a group of attentive colleagues seated around a large table. He is wearing a crisp white dress shirt and tailored gray trousers, exuding professionalism and poise. His hair is neatly styled, and he gestures smoothly with his hands to emphasize his points, creating a dynamic and engaging scene. The colleagues, consisting of both men and women, are focused on him, some taking notes. The room is well-lit, with a large digital screen behind him displaying key points of his presentation. The camera pans slowly across the room, capturing the interaction and the attentive atmosphere.',
    'A charming woman is washing dishes at a kitchen sink. She is in her late twenties, with long, wavy brown hair tied back, and wearing a light blue blouse with rolled-up sleeves. Her expression is content and relaxed as she focuses on her task. The kitchen is bathed in warm, natural light coming through a nearby window, adding a cozy atmosphere to the scene. The woman stands slightly angled towards the camera, with her hands immersed in soapy water, gently scrubbing a plate. The camera slowly zooms in to capture the details of her actions, highlighting the splashes of water and bubbles as she washes.',
    "A man is seated at an outdoor café, casually enjoying a burger. He is in his mid-30s with a neatly trimmed beard and a friendly expression, wearing a stylish navy blue jacket over a casual t-shirt. As he takes a bite, the camera captures the details of the juicy burger, with lettuce and tomato peeking out from the bun. Sunlight filters through the trees around the café, creating a pleasant, relaxed atmosphere. The camera angles slightly upward as it focuses on the man's content expression, emphasizing his enjoyment of the meal.",
    'A person, bundled in a heavy winter coat with a hood, is trudging through a snowstorm. Snowflakes swirl around them, blurring the background of snow-covered trees. They move slowly, facing towards the camera, with their head slightly bowed against the biting wind. The atmosphere is cold and wintry, with a sense of perseverance as the person navigates through the thick flurries. The camera provides a steady, close view, emphasizing the intensity of the snowstorm and the determination of the person.',
    # object_class
    'A young and beautiful woman is casually sitting on a park bench surrounded by lush greenery. She has long, flowing hair and is wearing a stylish, light summer dress with floral patterns. Her eyes are bright and animated as she gazes thoughtfully into the distance, with a gentle smile gracing her lips. Her posture is relaxed, with one leg crossed over the other. The scene is bathed in warm, natural light, creating a serene and peaceful atmosphere. The camera smoothly pans from right to left to capture her elegant profile against the backdrop of the vibrant park setting.',
    "A sleek, modern bicycle rests upright on a paved path surrounded by lush greenery. The bicycle has a metallic frame with a matte finish, sleek black tires, and a comfortable saddle. Sunlight filters through the leaves of the trees above, casting dappled patterns of light and shadow on the path. The camera pans slowly from the front wheel to the back, capturing the bicycle's streamlined design and the tranquil setting.",
    "A sleek, red sports car is parked at an angle on a sunlit road, surrounded by vibrant green trees in the background. The car's glossy surface reflects the sunlight, accentuating its smooth curves and stylish design. The windows are slightly tinted, and the polished chrome wheels add a touch of luxury. The camera slowly pans from the front to the side, showcasing the car's elegant profile. The overall atmosphere is bright and serene, highlighting the car as the focal point.",
    'A sleek motorcycle is parked on a quiet, open road surrounded by a picturesque natural landscape. The motorcycle has a shiny black and chrome finish that reflects the ambient light, highlighting its elegant curves. The road stretches into the distance, bordered by lush green fields and a blue sky above. The atmosphere is peaceful, with a gentle breeze suggested by the movement of nearby grass. The camera slowly pans from left to right, capturing the motorcycle from various angles and emphasizing its polished design.',
    "A sleek, modern airplane is captured from a side view as it soars through a clear blue sky. The sunlight reflects off its polished metal surface, emphasizing the plane's streamlined design. The airplane's wings are slightly tilted, indicating a gentle ascent. Contrails form behind its engines, adding to the sense of motion and speed. The camera pans slowly from left to right, following the airplane's graceful movement against the vast sky.",
    # ------ from other papers ------
    # InfinityStar
    'A woman with shoulder-length brown hair is seen talking to someone off-screen to the right. She is wearing a dark-colored top and a necklace. The background is blurred, but it appears to be an indoor setting with some indistinct objects and a window. The woman slightly moves her head while speaking.',
    'A man wearing a green Adidas shirt and a blue cap is holding a sketchbook with both hands. He is standing outdoors in front of a building with a blue facade and some greenery. The man is flipping through the pages of the sketchbook. He points to different parts of the drawings while talking.',
    'The camera gently pans over the mountainside, gliding above the treetops, and offering an expansive view of the mountains and the distant lake. As the drone flies smoothly, the entire natural landscape unfolds without abrupt perspective shifts, presenting the audience with a wide view and a feeling of tranquility.',
    # 
    'A video shows a peaceful snow - covered forest with tall pines. A silver BMW with headlights on is parked on a snowy path, its "X054TP 799" license plate visible. The warm headlight glow contrasts the cold snow, and the fixed camera emphasizes the serene winter scene.',
    'A video shows a woman singing on stage. In dark T-shirt with a graphic, necklace and black earrings, she holds a microphone to her cheek, with subtle posture and expression changes. Dimly lit with a curtain in the background, the fixed camera focuses on her, creating an intimate atmosphere.',
    'The video shows a panda hanging from thick ropes in what seems an indoor zoo enclosure with rocks, trees and bright lights. It makes diverse flexible, playful movements, alternating hand grips, moving legs, pushing its body down, reaching for rocks and lifting legs. Its black - and - white fur contrasts sharply with the natural background, and it looks calm and joyful as the camera tracks it.',
    'A video shows a man in a white chef’s uniform in a modern kitchen. The cluttered counter has various utensils and food (likely pizza). With a “GOD Bless AMERICA” sign on the wall, he takes a fork and knife, then cuts the food, looking focused. Bright lights and a fixed - perspective camera highlight the scene.',
    'An aerial video shows a stunning mountain range with jagged, layered eroded rock columns. Light - colored rocks contrast with sparse green vegetation on the dry hillside, and distant hills and valleys form a layered landscape. The clear bright blue sky enhances the serene yet imposing natural grandeur.',
    # 
    'A man stands in a well-lit kitchen with white cabinets, a large window, and various kitchen items on the counter and shelves. He gestures with his hand while speaking, then turns and walks towards a pot on the stove. The man reaches out to lift the lid of the pot.',
    'The video shows a cable car system with a tower in the foreground, situated on a mountainous area with lush greenery. In the background, there is a cityscape with buildings and a river, partially obscured by fog. A cable car moves from the right towards the left.',
    # 
    'A white owl glides smoothly through the warm desert air at golden hour. Its wings move slowly and powerfully, catching the sunlight as it soars above the sand and cacti. The camera follows its graceful motion, capturing the serene, cinematic beauty of its flight against the glowing horizon.',
    'A dynamic low-angle shot of a snowboarder carving sharply on a mountain slope, in sleek futuristic sportswear. The dramatic black-and-white scene has swirling fog, sparse clouds and subtle film grain, with moody diffuse lighting accentuating snow contours and the snowboarder’s pose.',
    'In a bright living room with green trees outside the window, the influencer praises the sofa for its comfort and style, ending with sipping coffee and a thumbs-up. The handheld iPhone footage, with smooth shot transitions, uses natural warm light and soft ambient sounds to convey an authentic and cozy feel.',
    'Ultra-realistic macro of a transparent iridescent glass apple with green glass stem and leaf, placed on soft pink marble table with gold veins, glowing pastel reflections of pink, blue, and gold, diffused daylight, blurred beige background, elegant minimal composition, HDR lighting, 9:16',
    # WAN
    'A Viking warrior wields a great axe with both hands, battling a mammoth at dusk, amidst a snowy landscape with snowflakes swirling in the air.',
    'An epic battle scene, unfolds as a tall and muscular Viking warrior wields a heavy great axe with both hands, facing off against a massive mammoth. The warrior is clad in leather armor and a horned helmet, with prominent muscles and a fierce, determined expression. The mammoth is covered in long hair, with sharp tusks, and roars angrily. It is dusk, and the snowy landscape is filled with swirling snowflakes, creating an intense and dramatic atmosphere. The backdrop features a barren ice field with the faint outlines of distant mountains. The use of cool-toned lighting emphasizes strength and bravery. The scene is captured in a dynamic close-up shot from a high-angle perspective.',
    'The camera follows a motorboat chasing dolphins in the sea.',
    'In a documentary photography style, the camera follows a motorboat chasing a pod of dolphins leaping out of the vast ocean. On the motorboat, there is a driver wearing a life jacket and a safety helmet, with a focused and excited expression. The dolphins are sleek and smooth-skinned, occasionally leaping out of the water and diving back in with agility. The sky is bright blue, the sun is shining, and the sea is shimmering with light. A few white clouds dot the distant sky. The scene is dynamic and full of energy, captured from a medium shot in a tracking perspective.',
    'The tiny Potato King, wearing a majestic crown, sits on the throne as potato subjects pay homage to it.',
    'In a surrealist style, the tiny Potato King wears a magnificent gold crown and sits on a towering throne. Its skin has a natural earthy yellow tone with subtle bumpy textures. The potato subjects are lined up on either side, bowing their heads in homage to the king. The background features the grand interior of a palace, with gold and red decorations that appear luxurious and solemn. A beam of light shines down from above, creating a sacred atmosphere. The scene is captured in a close-up shot from a high-angle perspective.',
    # 
    "Retro 80s Monster Horror Comedy Movie Scene: Color film, children's bedroom bathed in soft, warm light. Plush monsters of various sizes and colors are having a chaotic party, jumping on the bed, dancing to upbeat music, and throwing confetti. The walls are adorned with posters of classic 80s movies, and the room is filled with the playful laughter of children.",
    "A sepia-toned vintage photograph depicting a whimsical bicycle race featuring several dogs wearing goggles and tiny cycling outfits. The canine racers, with determined expressions and blurred motion, pedal miniature bicycles on a dusty road. Spectators in period clothing line the sides, adding to the nostalgic atmosphere. Slightly grainy and blurred, mimicking old photos, with soft side lighting enhancing the warm tones and rustic charm of the scene. 'Bicycle Race' captures this unique moment in a medium shot, focusing on both the racers and the lively crowd.",
    'Film quality, professional quality, rich details. The video begins to show the surface of a pond, and the camera slowly zooms in to a close-up. The water surface begins to bubble, and then a blonde woman is seen coming out of the lotus pond soaked all over, showing the subtle changes in her facial expression, creating a dreamy atmosphere.',
    'Sports photography full of dynamism, several motorcycles fiercely compete on the loess flying track, their wheels rolling up the dust in the sky. The motorcyclist is wearing professional racing clothes. The camera uses a high-speed shutter to capture moments, follows from the side and rear, and finally freezes in a close-up of a motorcycle, showcasing its exquisite body lines and powerful mechanical beauty, creating a tense and exciting racing atmosphere. Close up dynamic perspective, perfectly presenting the visual impact of speed and power.',
    'A professional male diver performs an elegant diving maneuver from a high platform. Full-body side view captures him wearing bright red swim trunks in an upside-down posture with arms fully extended and legs straight and pressed together. The camera pans downward as he dives into the water below, creating a dramatic splash with perfect entry form.',
    "A weightless young man, with soft features and an expression of serene astonishment, is slowly drifting above a sun-drenched field of swaying grass. Filmed in a retro cinematic style with warm golden tones and slight grain, side light accentuates the texture of his tousled hair as the wind gently brushes through it. The wide-angle lens captures the expansive landscape, enhancing the quiet, surreal levitation scene filled with calm wonder and gentle absurdity, defying reality's expectations.",
    'A distinctive father-son duo rides bicycles through city streets, clad in eye-catching attire - the father in a vibrant red suit and the son in neat school uniform. Their striking feature: giant yellow balloons replacing their heads, each meticulously painted with celebratory Chinese character "囍" in bold black ink, gently swaying in the wind as they pedal through the urban landscape.',
    'A retro 70s-style title sequence for a fictional action movie. Hand-drawn, stylized text "WAN" appears dynamically on screen, overlaid on fast-paced clips of car chases, explosions, and daring stunts. The text is bold, gritty, and slightly distorted, reflecting the 70s action movie aesthetic. A montage of high-octane scenes with a retro film grain effect, featuring warm, vintage colors. The sequences are bathed in golden hour light, enhancing the nostalgic feel.',
    #
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
