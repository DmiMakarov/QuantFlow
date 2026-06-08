import base64
from pathlib import Path

# Convert image to base64]\

for i in range(10, 11):
    image_path = f'{i}.jpg'
    image_data = base64.b64encode(Path(image_path).read_bytes()).decode()

    with open(f'{i}.txt', 'w') as f:
        f.write(f'<img src="data:image/png;base64,{image_data}" alt="Image">')
