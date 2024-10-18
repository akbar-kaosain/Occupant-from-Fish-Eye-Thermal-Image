import cv2
from fisheyewarping import FisheyeWarping

try:
    # Load the fisheye image
    fisheye_img = cv2.imread('C:/Users/syrin/OneDrive/Bureau/work-desJardins/Multiple_people_SC2/output92.jpg')
    if fisheye_img is None:
        raise ValueError("Image not found or unable to load.")

    # Initialize the FisheyeWarping class
    frd = FisheyeWarping(fisheye_img, use_multiprocessing=True)

    # Build and save the dewarp mesh
    frd.build_dewarp_mesh(save_path='C:/Users/syrin/OneDrive/Bureau/work-desJardins/Multiple_people_SC2/dewarp-mesh.pkl')

    # Run dewarping and save output
    frd.run_dewarp(save_path='C:/Users/syrin/OneDrive/Bureau/work-desJardins/Multiple_people_SC2/dewarp-output92.jpg')

    print("Dewarping completed successfully!")

except Exception as e:
    print(f"An error occurred: {e}")