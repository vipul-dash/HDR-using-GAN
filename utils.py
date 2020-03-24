import numpy as np 
import math
def weights(pixel_values):
#the weighting function for calculating weights from pixels
 zmin,zmax=0,255

 z=np.random.randint(zmin,zmax)


 zmid= (zmin+zmax)//2

 if (z < zmid):
    w=z-zmin
 else:

    w=zmax-z

def sample_pixels(images):
     # sampling the pixels from images 
      zmin,zmax=0,255

      num_images=len(images)
      num_intensities=zmax-zmin+1

      intensites=np.zeros((num_intensities,num_images),dtype=np.int64)
      mid_img=images[num_images//2]
      for i in range(zmin,zmax+1):
        rows,cols=np.where(mid_img==i)
         
        if(len(rows)!=0):
            index=np.random.randint(len(rows))
            for j in range(num_images):
                intensites[i,j]=images[j][rows[idx],cols[idx]]
      return intensites

def compute_response_curve(intensites,log_exposure_times,weighting_func,smooth_lambda ): 
# get the response curve 
    zmin,zmax=0,255
    num_images=len(log_exposure_times)
    num_samples=len(intensites.shape[0])
    intensity_range=zmax-zmin
    mat_A=np.zeros((num_samples*num_images+intensity_range,num_samples+intensity_range+1),dtype=np.int64)
    mat_B=np.zeros((mat_A.shape[0],1),dtype=np.int64)
    #data-fitting 
    k=0 
    
    for i in range(num_samples):
        for j in range(num_images):
            zij=intensites[i,j]
            wij=weighting_func(zij)
            mat_A[k,zij]=wij
            mat_A[k,(intensity_range+1)+i]=-wij
            mat_B[k,0]= wij*log_exposure_times[j]
            k+=1
    

    for zk in range(zmin+1,zmax):
            wzk=weighting_func(zk)
            
            mat_A[k,zk-1]=wzk*smooth_lambda
            mat_A[k,zk]=-2*wzk*smooth_lambda
            mat_A[k,zk+1]=wzk*smooth_lambda
            k+=1




    mat_Ainv=np.linalg.pinv(mat_A)

    x=np.dot(mat_Ainv,mat_B)
    g=x[0:intensity_range+1]

    return g[:,0]



def radiance_map(images,compute_response_curve,weighting_func,log_exposure_times):
 #get the radiance map from the response curve
  num_images_=len(log_exposure_times)
  img_shape =images[0].shape
  image_radiance_map =np.zeros(img_shape,dtype=np.float64)
  
  for i in range(img_shape[0]):
      for j in  range(img_shape[1]):
          

          g = np.array([compute_response_curve(images[k][i,j]) for k in range(num_images) ])
          w=  np.array([weighting_func(images[k][i,j] )for k in range(num_images)   ])

          weight_sum=np.sum(w)
          if sum > 0: 
            image_radiance_map[i,j]=  np.sum(w* (g - log_exposure_times)/weight_sum)
          else :
            image_radiance_map[i,j]=np.sum(g[num_images//2] - log_exposure_times[num_images//2])            
  return image_radiance_map  


def local_tone_mapping(image,gamma):
     # gamma is an 2d array of the same shape as of image and contains the gamma values for each pixels
     #http://cs.brown.edu/courses/cs129/results/proj5/njooma/
     #https://en.wikipedia.org/wiki/Tone_mapping
     rows,cols=image.shape[0],image.shape[1]
     assert image.shape==gamma.shape
     for i in range(rows):
         for j in range(cols):

             image[i][j]=math.pow(image[i][j],gamma[i][j])

     # each pixel is tone mapped differently accoring to location         
             
     

    


             
             
        
       


         

      
         







