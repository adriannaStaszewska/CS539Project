import numpy as np
import cv2

def generate_random_lines(imshape,slant,drop_length):    
	
	drops=[]    
	for i in range(1000): 
	## If You want heavy rain, try increasing this        
		if slant<0:            
			x= np.random.randint(slant,imshape[1])        
		else:            
			x= np.random.randint(0,imshape[1]-slant)        
			y= np.random.randint(0,imshape[0]-drop_length)        
			drops.append((x,y))    
	
	return drops            
	
def add_rain(image):        

	imshape = image.shape    
	slant_extreme=10    
	slant= np.random.randint(-slant_extreme,slant_extreme)     
	drop_length=20    
	drop_width=2    
	drop_color=(200,200,200) 
	## a shade of gray    
	rain_drops= generate_random_lines(imshape,slant,drop_length)        
	
	for rain_drop in rain_drops:        
		cv2.line(image,(rain_drop[0],rain_drop[1]),(rain_drop[0]+slant,rain_drop[1]+drop_length),drop_color,drop_width)    
	
	image= cv2.blur(image,(3,3)) 
	## rainy view are blurry        
	
	brightness_coefficient = 0.7 
	## rainy days are usually shady     
	
	image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS) 
	## Conversion to HLS    
	
	image_HLS[:,:,1] = image_HLS[:,:,1]*brightness_coefficient 
	## scale pixel values down for channel 1(Lightness)    
	
	image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB) 
	## Conversion to RGB    
		
	return image_RGB

if __name__ == "__main__":
	image = cv2.imread("../../Downloads/photo.jpg")
	new_img = add_rain(image)
	print(new_img.shape)
	cv2.imwrite('../../Desktop/img.jpg', new_img)

