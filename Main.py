#Initial Commit
import torch
import numpy as np


print("hello world")

x = torch.rand(5,3) #dimensionality is like this first value is rows, second, columns, third depth. Same as xyz
                    # dimensions

ones = torch.ones(6,4)
zeros = torch.zeros(7,5)

mybsarray=([1,2,3,0])

mybsarray_tensor=torch.tensor(mybsarray) #use to convert array data to tensor

#numpy array to tensor - possible use if importing CSV data as np array
npones = np.ones(5)
tnpones = torch.from_numpy(npones)


print(x)
print(ones)
print(zeros)
print(mybsarray_tensor)
print(f"this is the tensor created from numpy ones {tnpones}")


#attributes of a tensor
print(f"Shape of a tensor: {ones.shape}")
print(f"data type of tensor: {zeros.dtype}")
print(f"Device where tensor is stored: {mybsarray_tensor.device}")


#determine if GPU is available
print(f"GPU processing available: {torch.cuda.is_available()} ")




