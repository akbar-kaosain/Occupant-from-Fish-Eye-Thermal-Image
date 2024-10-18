#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 13:56:31 2024

@author: atelier
"""

#!/usr/bin/env python
'''
import sys
import numpy as np
import cv2
from pylepton import Lepton

def capture(flip_v = False, device = "/dev/spidev0.0"):
  with Lepton(device) as l:
    a,_ = l.capture()
  if flip_v:
    cv2.flip(a,0,a)
  cv2.normalize(a, a, 0, 65535, cv2.NORM_MINMAX)
  np.right_shift(a, 8, a)
  return np.uint8(a)

if __name__ == '__main__':
  from optparse import OptionParser

  usage = "usage: %prog [options] output_file[.format]"
  parser = OptionParser(usage=usage)

  parser.add_option("-f", "--flip-vertical",
                    action="store_true", dest="flip_v", default=False,
                    help="flip the output image vertically")

  parser.add_option("-d", "--device",
                    dest="device", default="/dev/spidev0.0",
                    help="specify the spi device node (might be /dev/spidev0.1 on a newer device)")

  (options, args) = parser.parse_args()
  print(options)
  if len(args) < 1:
    print("You must specify an output filename")
    sys.exit(1)

  image = capture(flip_v = options.flip_v, device = options.device)
  
  cv2.imwrite(args[0], image)
  '''
########################################
'''
import numpy as np
import cv2
import sys
from pylepton import Lepton3

import time
time.sleep(2)
with Lepton3() as l:
  x=time.time()
  for i in range (200):
      
      print(i)
      a,_ = l.capture()  
      cv2.normalize(a, a, 0, 65536, cv2.NORM_MINMAX) # extend contrast
      
      np.right_shift(a, 8, a) # fit data into 8 bits
      #cv2.imshow("reza", np.uint8(a))
      cv2.imwrite(f"output{i}.jpg", np.uint8(a)) # write it
print(time.time()-x)'''

''' 
import numpy as np
import cv2
import sys
from pylepton import Lepton3

with Lepton3() as l:
    while True:
        # Capture a frame from the thermal camera
        a, _ = l.capture()
        print("Frame captured")

        # Normalize the captured frame to extend contrast
        cv2.normalize(a, a, 0, 65536, cv2.NORM_MINMAX)
        
        # Fit data into 8 bits
        np.right_shift(a, 8, a)
        
        # Convert to uint8
        frame = np.uint8(a)
        
        # Display the frame in a window
        cv2.imshow('Thermal Camera Feed', frame)

        # Check for 'q' key press to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Clean up and close the display window
cv2.destroyAllWindows()
'''

import numpy as np
import cv2
from pylepton import Lepton3
import time
import os

# Create the output directory if it doesn't exist
output_directory = "Multiple_people_SC6"
os.makedirs(output_directory, exist_ok=True)

time.sleep(5)

with Lepton3() as l:
    start_time = time.time()
    for i in range(500):
        print(i)
        a, _ = l.capture()  
        cv2.normalize(a, a, 0, 65536, cv2.NORM_MINMAX)  # Extend contrast
        
        np.right_shift(a, 8, a)  # Fit data into 8 bits
        
        # Construct the filename with the output directory
        filename = os.path.join(output_directory, f"output{i}.jpg")
        
        # Write the image to the specified directory
        cv2.imwrite(filename, np.uint8(a))  # Write it
    
    print(f"Total time taken: {time.time() - start_time:.2f} seconds")
