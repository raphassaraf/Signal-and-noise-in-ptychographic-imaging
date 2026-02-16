from src.ptychography import simulate_object


PATH = '64_64_img.jpg'
OBJ_SIZE = 64
TRUE_IMG = simulate_object(PATH, resize=OBJ_SIZE)