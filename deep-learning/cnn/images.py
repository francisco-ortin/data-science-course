# some images for image classification
image_URLs = [
   'https://images.unsplash.com/photo-1560807707-8cc77767d783',
   'https://images.unsplash.com/photo-1518709268805-4e9042af9f23',
   'https://images.unsplash.com/photo-1552519507-da3b142c6e3d',
   'https://images.unsplash.com/photo-1527549993586-dff825b37782',
   'https://images.unsplash.com/photo-1555041469-a586c61ea9bc',
   'https://images.unsplash.com/photo-1517336714731-489689fd1ca8',
   'https://images.unsplash.com/photo-1511707171634-5f897ff02aa9',
   'https://images.unsplash.com/photo-1567306226416-28f0efdc88ce',
   'https://images.unsplash.com/photo-1546069901-ba9599a7e63c',
   'https://images.unsplash.com/photo-1506748686214-e9df14d4d9d0',
   'https://images.unsplash.com/photo-1507525428034-b723cf961d3e',
   'https://images.unsplash.com/photo-1500530855697-b586d89ba3ee'
    ]


# plane images for object detection
plane_image_URLS = [
    'https://irp.cdn-website.com/e346530e/dms3rep/multi/airplanes.png',
    'https://images.stockcake.com/public/9/9/4/994c4c96-99ca-4cce-90ce-74a714af9f4a_large/sky-high-traffic-stockcake.jpg',
    'https://www.travelandleisure.com/thmb/Nq9fBPWYGxNEmUvkK3P1b96F7XU=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/composite-heathrow-airport-PLANEINSKY0322-f4e3d471b6b64c84bb297e5e7347076e.jpg'
    ]

# Image segmentation

# Define labels and colors for each class
class_labels = {
    0: "Background", 1: "Hat", 2: "Hair", 3: "Sunglasses", 4: "Upper-clothes",
    5: "Skirt", 6: "Pants", 7: "Dress", 8: "Belt", 9: "Left-shoe", 10: "Right-shoe",
    11: "Face", 12: "Left-leg", 13: "Right-leg", 14: "Left-arm", 15: "Right-arm",
    16: "Bag", 17: "Scarf"
}

# define color for each class
color_mapping = {
    0: (0, 0, 0),          # Background - black
    1: (255, 0, 0),        # Hat - red
    2: (255, 165, 0),      # Hair - orange
    3: (255, 255, 0),      # Sunglasses - yellow
    4: (0, 128, 0),        # Upper-clothes - green
    5: (0, 255, 0),        # Skirt - bright green
    6: (0, 0, 255),        # Pants - blue
    7: (75, 0, 130),       # Dress - indigo
    8: (238, 130, 238),    # Belt - violet
    9: (128, 0, 0),        # Left-shoe - maroon
    10: (139, 69, 19),     # Right-shoe - brown
    11: (255, 182, 193),   # Face - pink
    12: (64, 224, 208),    # Left-leg - turquoise
    13: (70, 130, 180),    # Right-leg - steel blue
    14: (100, 149, 237),   # Left-arm - cornflower blue
    15: (147, 112, 219),   # Right-arm - medium purple
    16: (255, 20, 147),    # Bag - deep pink
    17: (0, 255, 255)      # Scarf - cyan
}

segmentation_image_URLs = [
    "https://plus.unsplash.com/premium_photo-1673210886161-bfcc40f54d1f?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8MXx8cGVyc29uJTIwc3RhbmRpbmd8ZW58MHx8MHx8&w=1000&q=80",
    "https://images.unsplash.com/photo-1643310325061-2beef64926a5?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8Nnx8cmFjb29uc3xlbnwwfHwwfHw%3D&w=1000&q=80",
    "https://freerangestock.com/sample/139043/young-man-standing-and-leaning-on-car.jpg"
    ]



