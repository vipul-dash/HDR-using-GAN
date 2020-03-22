import numpy as np 

def weights(pixel_values):

 zmin,zmax=0,255

 z=np.random.randint(zmin,zmax)


 zmid= (zmin+zmax)//2

 if (z < zmid):
    w=z-zmin
 else:

    w=zmax-z

def sample_pixels(images):
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




    mat_A+=np.linalg.pinv(mat_A)

    x=np.dot(mat_A+,mat_B)
    g=x[0:intensity_range+1]

    return g[:,0]

    
     

    


             
             
        
       


         

      
         







