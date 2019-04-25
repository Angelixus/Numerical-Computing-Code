from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
#%% I <-- means Image,   a <- means numpy array
I = Image.open('lena.jpg') # Open the image with name 'lena.jpg' using the Image object
a = np.asarray(I) # Transform the image to a numpy array, this array is a matrix that has "3 layers" (RGB), each element is an unsigned int with 8 bits [0, 255]
plt.imshow(a)
plt.colorbar()
plt.show()


I1 = I.convert('L') # Transform the image to black and white
a1 = np.asarray(I1, dtype=np.float64) # Transform the image to an array, dtype lets us define the data type (from 8 bit uint to double)
plt.imshow(a1, cmap='gray') # cmap lets us define the color map, so  we can show the same image in different ways
plt.colorbar() # Show the map referring to the colormap some colors means high values (close to 255) other mean low values (close to 0)
plt.show()
#%% Compute gradient of I1 (a1)
gy, gx = np.gradient(a1) # Compute the gradient of the I1 image array representation
a2 = -np.sqrt(gx**2+gy**2) # We invert the image with the - (we have a different range of values that [0, 255])
plt.imshow(a2, cmap='gray') # cmap lets us define the color map, so  we can show the same image in different ways
plt.colorbar() # Show the map referring to the colormap some colors means high values (close to 255) other mean low values (close to 0)
plt.show()
"""
Steps:
1. Convert our range in a (original array) to range [0, x] substracting the array minus its minimum value and store it on a new array (a1 for instance)
2. Convert the range in a1 to range [0, 1] dividing by the difference of max a min values of a (the original array)
3. Convert the range in a1 to range [0, 255] multiplying by 255 the a1 array
4. Change, if needed, the data type of a1 into uint8 and save it on a new array a2 
"""
#%% Save image as .png, convert our image to the range  [0, 255] using uint
a3 = a2 - np.min(a2) # This is in order to set the minimum value of the image to 0 so have the range start at 0, not like before
a3 = a3 / (np.max(a2) - np.min(a2)) # Make the range of our image in [0, 1] (plt does the interpolation regardless of ranges, so no problem), since we apply this "filters" to every number in the array, the image does not change
a3 *= 255 # Transform the previous range [0, 1] to [0, 255]
a4 = a3.astype(np.uint8) # Change the data type from double to uint with 8 bits
plt.imshow(a4, cmap='gray') # cmap lets us define the color map, so  we can show the same image in different ways
plt.colorbar() # Show the map referring to the colormap some colors means high values (close to 255) other mean low values (close to 0)
plt.show()
I2 = Image.fromarray(a4) # Convert the array to image
I2.save('lena_mod.png') # Save the image