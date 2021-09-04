from keras.models import load_model
from tensorflow.keras.preprocessing import image
#Load Saved Model File
model=load_model(r'C:\Users\Antarlin\Desktop\Data_Science\Deployment26July2021\Fire Detection\flask_app\firedetection.h5')

# image path
# input  image location that is to be classified
img_path=r'C:\Users\Antarlin\Desktop\Data_Science\Deployment26July2021\Fire Detection\fire_dataset\img_to_verify\fire.28.png'

# read the image
test_image=image.load_img(img_path,target_size=(64,64))


# image to array
test_image=image.img_to_array(test_image)
test_image=test_image.reshape(1,64,64,3)
result=model.predict(test_image)
print(result)

if result[0][0]==0:
    print("Fire")
else:
    print("No Fire")