import numpy as np
import cv2
from matplotlib import pyplot as plt

'''
This sample script takes in the file path to a 16-bit distorted tiff captured from a Lepton 3.1R or 
Lepton UW as the input and creates a LeptonDewarp object which is used to define transformation 
matrices for dewarping a Lepton 3.1R or Lepton UW image. Then, it applies the properties from the
LeptonDewarp object and calls the get_undistorted_img() function to correct the input image. Finally,
the script displays the original distorted image and its corrected version side-by-side. 

The LeptonDewarp class defines how to correct distorted images by setting up the transformation 
matrices and calling the built-in OpenCV functions for the correction.

'''
class LeptonDewarp:
    '''
        Undistorts images captured with Lepton 3.1R(WFOV95) or Lepton UW(WFOV160).
    '''
    
    # Constant nested dictionary that stores the camera matrix, distortion coefficients, and new camera matrix for each WFOV
    camera_parameters = {'WFOV95': { 'camera matrix': [[104.65403680863373, 0.0, 79.12313258957062],
                                                      [0.0, 104.48251047202757, 55.689070170705634],
                                                      [0.0, 0.0, 1.0]],
                                    'distortion coeff': [[-0.39758308581607127,
                                                          0.18068641745671193,
                                                          0.004626461618389028,
                                                          0.004197358204037882,
                                                          -0.03381399499591463]],
                                    'new camera matrix':[[66.54581451416016, 0.0, 81.92717558174809],
                                                             [0.0, 64.58526611328125, 56.23740168870427], 
                                                             [0.0, 0.0, 1.0]]},
                         'WFOV160': { 'camera matrix': [[52.22140633962198, 0.0, 77.47503198855625],
                                                        [0.0, 50.96485821522407, 60.98233009842293],
                                                        [0.0, 0.0, 1.0]],
                                      'distortion coeff': [[-0.017183577427603652,
                                                            -0.0529287336515663,
                                                            0.01892571114679248,
                                                            -0.0020057923994113904]],
                                      'new camera matrix': [[13.954273635851617, 0.0, 73.45370485973413],
                                                            [0.0, 13.618506800113208, 63.225336170962834],
                                                            [0.0, 0.0, 1.0]]}}
   
    def __init__(self, wfov):
        '''
            LeptonDewarp Constructor that initializes the matrices for dewarping the image
            that corresponds to the field-of-view

            Args:
                wfov = string,
                    field-of-view of the camera, "WFOV95" or "WFOV160"
        '''
        
        self.wfov = wfov

        # Get camera matrix,distortion coefficients, and new camera matrix according to the field of view
        self.camera_matrix = np.array(LeptonDewarp.camera_parameters.get(self.wfov, {}).get('camera matrix'))
        self.distortion_coeff = np.array(LeptonDewarp.camera_parameters.get(self.wfov, {}).get('distortion coeff'))
        self.new_camera_matrix = np.array(LeptonDewarp.camera_parameters.get(self.wfov, {}).get('new camera matrix'))


    def get_undistorted_img(self, img, retain_pixels=False, crop=False):
        '''
            Undistort the image

            Args:
                img = numpy array,
                    distorted image in uint8

                cretain_pixels = boolean,
                    default to False, dewarp image to have a fixed IFOV
                    True will keep all black border pixels after dewarped
                
                crop = boolean,
                    only applies when retain_pixels is True;
                    crops image to remove black pixels
            
            Output:
                undistorted_img = numpy array,
                    corrected image;
                    same resolution as input if crop isn't applied
        '''

        if self.wfov == 'WFOV95':
            # Apply matrices to undistort function to correct image
            if retain_pixels:
                # Keep all pixels from input after dewarp
                undistorted_img = cv2.undistort(img, self.camera_matrix,
                                                self.distortion_coeff,
                                                None,
                                                self.new_camera_matrix)

                if crop:
                    # Get image dimension
                    img_dim = undistorted_img.shape
                    row = img_dim[0]
                    col = img_dim[1]

                    # OpenCV generated cropping matrix still retains a few black pixels,
                    # return the corrected image with those pixels cropped out
                    undistorted_img = undistorted_img[14:row-18, 12:col-12]
            else:
                # Remove borders after dewarp
                undistorted_img = cv2.undistort(img,
                                                self.camera_matrix,
                                                self.distortion_coeff)

        elif self.wfov == 'WFOV160':
            # Apply matrices to undistort function to correct image
            if retain_pixels:
                # Keep all pixels from input after dewarp
                undistorted_img = cv2.fisheye.undistortImage(img,
                                                             K=self.camera_matrix,
                                                             D=self.distortion_coeff,
                                                             Knew=self.new_camera_matrix)
                
                if crop:
                    # Get image dimension
                    img_dim = undistorted_img.shape
                    row = img_dim[0]
                    col = img_dim[1]

                    # OpenCV generated cropping matrix still retains a few black pixels,
                    # return the corrected image with those pixels cropped out
                    undistorted_img = undistorted_img[2:row-13, :]
            else:
                # Remove borders after dewarp
                undistorted_img = cv2.fisheye.undistortImage(img,
                                                             K=self.camera_matrix,
                                                             D=self.distortion_coeff,
                                                             Knew=self.camera_matrix)

        return undistorted_img

def convert_raw_img(img_in):
    '''
        Normalizes and converts a uint16 raw image captured from a Lepton into uint8

        Args: 
            img_in = numpy array

        Output: 
            img = numpy array,
                  image in uint8
    '''

    img = cv2.normalize(img_in, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
    img = (img/256).astype('uint8')
    
    return img

# Sample Code applying the LeptonDewarp Class
if __name__ == "__main__":

    # Define file path and file name to distorted image and read it
    img_path = "output29.jpg"
    print("passing the image shape")
    img = cv2.imread(img_path, -1)
    img = convert_raw_img(img)

    # Create an object to get the matrices that corresponds to the fov
    cam = LeptonDewarp("WFOV160")

    # Apply distortion correction on input image
    undistorted_img = cam.get_undistorted_img(img, True)

    # Display original and corrected images side by side
    fig = plt.figure()
    orig = fig.add_subplot(1, 2, 1)
    print("imshow")
    orig.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    orig.set_title("original")
    plt.axis('off')
    result = fig.add_subplot(1, 2, 2)
    result.imshow(cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB))
    result.set_title("undistorted")
    plt.axis('off')
    plt.show()